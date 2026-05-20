import os
import pandas as pd
from typing import List, Literal, Optional
import httpx
from openai import OpenAI

from feature_agent import FeatureAgent
from prompts import (
    PREVIOUS_ERROR_TEMPLATE, 
    REFLECTION_PROMPT_TEMPLATE, 
    GENERATION_REFLECTED_PROMPT_TEMPLATE, 
    GENERATION_PROMPT_TEMPLATE
)

# Default proxy URL (can be overridden)
DEFAULT_PROXY_URL = "https://sberailab_proxy:NP1SPamfpiAvJ0wJ48td@scorpion1.ddns.net"

# OpenRouter configuration
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_DEFAULT_MODEL = "anthropic/claude-3.5-sonnet"


class OpenAIFeatureAgent(FeatureAgent):
    """FeatureAgent with OpenAI-compatible API integration.
    
    Works with OpenAI, OpenRouter, Azure OpenAI, and other compatible APIs.
    """
    
    def __init__(self, 
                 model_name: str = "gpt-4o",
                 api_key: Optional[str] = None,
                 api_key_env_var: str = "OPENAI_API_KEY",
                 base_url: Optional[str] = None,
                 proxy_url: Optional[str] = None,
                 use_proxy: bool = True,
                 temperature: float = 0.3,
                 max_tokens: int = 4096,
                 default_headers: Optional[dict] = None,
                 **kwargs):
        """Initialize OpenAI-compatible Feature Agent.
        
        Args:
            model_name: Model name (e.g., 'gpt-4o', 'gpt-4-turbo', 'gpt-3.5-turbo').
            api_key: API key. If None, uses environment variable.
            api_key_env_var: Environment variable name for API key.
            base_url: Optional custom base URL for API.
            proxy_url: Proxy URL for API requests. If None and use_proxy=True, uses default proxy.
            use_proxy: Whether to use proxy. Set to False to disable proxy. Default: True.
            temperature: Sampling temperature for generation.
            max_tokens: Maximum tokens for generation.
            default_headers: Optional default headers for API requests.
            **kwargs: Additional arguments passed to FeatureAgent.
        """
        # Get API key
        self.api_key = api_key or os.environ.get(api_key_env_var)
        if not self.api_key:
            raise ValueError(f"API key must be provided or set in {api_key_env_var} environment variable")
        
        self.base_url = base_url
        
        # Set up proxy
        if use_proxy:
            self.proxy_url = proxy_url or os.environ.get("OPENAI_PROXY_URL") or DEFAULT_PROXY_URL
        else:
            self.proxy_url = None
        
        # Build client kwargs
        client_kwargs = {"api_key": self.api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        if default_headers:
            client_kwargs["default_headers"] = default_headers
        
        # Add proxy via httpx client if specified
        if self.proxy_url:
            http_client = httpx.Client(proxy=self.proxy_url)
            client_kwargs["http_client"] = http_client
            
        self.client = OpenAI(**client_kwargs)
        self.model_name = model_name
        self._temperature = temperature
        self._max_tokens = max_tokens
        
        # Remove llm from kwargs since we're handling it separately
        kwargs.pop('llm', None)
        super().__init__(llm=None, **kwargs)
        
    def _generate_with_api(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate text using API."""
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self._temperature,
                max_tokens=self._max_tokens,
            )
            return response.choices[0].message.content
        except Exception as e:
            self.logger.error(f"API error: {e}")
            raise

    def _generate_feature_code(self) -> str:
        """Generate feature code using API with enhanced prompting."""

        def generate_code_or_prompt(prompt: str, mode: Literal["reflection", "generation"]) -> Optional[str]:
            system_prompt = (
                "You are an expert data scientist specializing in feature engineering. "
                "Generate clean, efficient Python code for creating features from transaction data. "
                "Always include necessary imports and handle edge cases."
            )
            
            generated_text = self._generate_with_api(prompt, system_prompt)
            
            if mode == "reflection":
                return generated_text
            else:
                code = self._extract_code_from_response(generated_text)
                if not code:
                    self.logger.warning("No code extracted from response")
                    return None
                return code

        try:
            if self.previous_error:
                enhanced_prompt = self._build_enhanced_prompt(None, None, None, "debugging")
                return generate_code_or_prompt(enhanced_prompt, "generation")
            elif self.reflection:
                base_context, previous_features_section = self._build_context()
                reflection_prompt = self._build_enhanced_prompt(
                    base_context, None, previous_features_section, "reflection"
                )
                generation_prompt = generate_code_or_prompt(reflection_prompt, "reflection")
                self.logger.info(f"Reflection output: {generation_prompt[:200]}...")
                enhanced_prompt = self._build_enhanced_prompt(
                    base_context, generation_prompt, None, "generation"
                )
                return generate_code_or_prompt(enhanced_prompt, "generation")
            else:
                base_context, previous_features_section = self._build_context()
                enhanced_prompt = self._build_enhanced_prompt(
                    base_context, None, previous_features_section, "generation"
                )
                return generate_code_or_prompt(enhanced_prompt, "generation")
        except Exception as e:
            self.logger.warning(f"Generation failed: {e}")
            return None
    
    def _build_enhanced_prompt(self, 
                               base_context: str, 
                               reflection_rules: str, 
                               previous_features_section: str, 
                               step: Literal["reflection", "generation", "debugging"]) -> str:
        """Build enhanced prompt with specific instructions."""
        if step == "reflection":
            return REFLECTION_PROMPT_TEMPLATE.format(
                base_context=base_context,
                previous_features_section=previous_features_section
            )
        elif step == "debugging":
            return PREVIOUS_ERROR_TEMPLATE.format(
                error_message=self.previous_error,
                client_id_col=self.client_id_col
            )
        elif step == "generation":
            if self.reflection:
                return GENERATION_REFLECTED_PROMPT_TEMPLATE.format(
                    base_context=base_context,
                    reflection_rules=reflection_rules,
                    client_id_col=self.client_id_col
                )
            else:
                return GENERATION_PROMPT_TEMPLATE.format(
                    base_context=base_context,
                    previous_features_section=previous_features_section,
                    client_id_col=self.client_id_col
                )
        else:
            raise ValueError(f"Invalid step: {step}")

    def batch_generate_features(self, n_features: int = 5) -> List[str]:
        """Generate multiple features sequentially."""
        successful_codes = []
        
        for i in range(n_features):
            try:
                code = self._generate_feature_code()
                if code:
                    successful_codes.append(code)
                    self.logger.info(f"Generated feature {i+1}/{n_features} successfully")
            except Exception as e:
                self.logger.warning(f"Feature {i+1} generation failed: {e}")
        
        return successful_codes

    def batch_execute_features(self, codes: List[str]) -> List[pd.DataFrame]:
        """Execute multiple feature codes in batch."""
        return self.code_engine.batch_execute_feature_codes(codes, self.sequential_train)


class OpenRouterFeatureAgent(OpenAIFeatureAgent):
    """FeatureAgent with OpenRouter API integration.
    
    OpenRouter provides access to many LLMs through a single API.
    See https://openrouter.ai/docs for available models.
    """
    
    def __init__(self, 
                 model_name: str = OPENROUTER_DEFAULT_MODEL,
                 api_key: Optional[str] = None,
                 site_url: Optional[str] = None,
                 site_name: Optional[str] = None,
                 **kwargs):
        """Initialize OpenRouter-based Feature Agent.
        
        Args:
            model_name: Model name from OpenRouter (e.g., 'anthropic/claude-3.5-sonnet', 
                       'openai/gpt-4o', 'meta-llama/llama-3.1-70b-instruct').
            api_key: OpenRouter API key. If None, uses OPENROUTER_API_KEY environment variable.
            site_url: Optional URL for OpenRouter rankings/analytics.
            site_name: Optional site name for OpenRouter rankings/analytics.
            **kwargs: Additional arguments passed to OpenAIFeatureAgent.
        """
        # Build OpenRouter-specific headers
        default_headers = {"X-Title": site_name or "TurboStats Feature Agent"}
        if site_url:
            default_headers["HTTP-Referer"] = site_url
        
        super().__init__(
            model_name=model_name,
            api_key=api_key,
            api_key_env_var="OPENROUTER_API_KEY",
            base_url=OPENROUTER_BASE_URL,
            default_headers=default_headers,
            **kwargs
        )


def test_api_connection(
    provider: str = "openai",
    api_key: Optional[str] = None,
    model_name: Optional[str] = None,
    proxy_url: Optional[str] = None,
    use_proxy: bool = True
) -> bool:
    """Test API connection with optional proxy.
    
    Args:
        provider: API provider ("openai" or "openrouter").
        api_key: API key. If None, uses environment variable.
        model_name: Model to use for test.
        proxy_url: Custom proxy URL. If None and use_proxy=True, uses default.
        use_proxy: Whether to use proxy.
    
    Returns:
        True if connection successful, False otherwise.
    """
    # Provider-specific configuration
    if provider == "openai":
        api_key_env = "OPENAI_API_KEY"
        base_url = None
        default_model = "gpt-4o"
        default_headers = None
    else:  # openrouter
        api_key_env = "OPENROUTER_API_KEY"
        base_url = OPENROUTER_BASE_URL
        default_model = OPENROUTER_DEFAULT_MODEL
        default_headers = {"X-Title": "TurboStats Feature Agent Test"}
    
    model = model_name or default_model
    
    print("=" * 60)
    print(f"{provider.upper()} API Connection Test")
    print("=" * 60)
    
    # Get API key
    api_key = api_key or os.environ.get(api_key_env)
    if not api_key:
        print("❌ ERROR: No API key found!")
        print(f"   Set {api_key_env} environment variable or pass --api-key")
        return False
    
    print(f"✓ API Key: {api_key[:8]}...{api_key[-4:]}")
    print(f"✓ Model: {model}")
    if base_url:
        print(f"✓ Base URL: {base_url}")
    
    # Set up proxy
    if use_proxy:
        effective_proxy = proxy_url or os.environ.get("OPENAI_PROXY_URL") or DEFAULT_PROXY_URL
        print(f"✓ Proxy: {effective_proxy[:30]}...")
    else:
        effective_proxy = None
        print("✓ Proxy: disabled")
    
    # Create client
    try:
        client_kwargs = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        if default_headers:
            client_kwargs["default_headers"] = default_headers
        if effective_proxy:
            http_client = httpx.Client(proxy=effective_proxy)
            client_kwargs["http_client"] = http_client
        
        client = OpenAI(**client_kwargs)
        print(f"✓ {provider.capitalize()} client created")
    except Exception as e:
        print(f"❌ Failed to create client: {e}")
        return False
    
    # Test API call
    print("\nSending test request...")
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"Say 'Hello, {provider.upper()} API is working!' in exactly those words."}
            ],
            max_tokens=50,
            temperature=0.0
        )
        
        result = response.choices[0].message.content
        print("\n✅ SUCCESS! Response received:")
        print(f"   {result}")
        print(f"\n   Model: {response.model}")
        if response.usage:
            print(f"   Tokens: {response.usage.total_tokens} (prompt: {response.usage.prompt_tokens}, completion: {response.usage.completion_tokens})")
        
        # Ask a question about feature engineering
        print("\n" + "-" * 60)
        print("Asking a question about feature engineering...")
        print("-" * 60)
        
        question = "What are 3 useful features you can create from bank transaction data to predict a customer's age? Give a brief answer."
        print(f"\n📝 Question: {question}\n")
        
        response2 = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a data scientist expert. Be concise."},
                {"role": "user", "content": question}
            ],
            max_tokens=300,
            temperature=0.7
        )
        
        answer = response2.choices[0].message.content
        print(f"🤖 Answer:\n{answer}")
        if response2.usage:
            print(f"\n   Tokens used: {response2.usage.total_tokens}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ API call failed: {e}")
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test OpenAI/OpenRouter API connection")
    parser.add_argument("--provider", type=str, choices=["openai", "openrouter"], default="openai",
                        help="API provider to test (default: openai)")
    parser.add_argument("--api-key", type=str, default=None,
                        help="API key (default: uses env var based on provider)")
    parser.add_argument("--model", type=str, default=None,
                        help="Model to use (default: gpt-4o for OpenAI, claude-3.5-sonnet for OpenRouter)")
    parser.add_argument("--proxy", type=str, default=None,
                        help="Custom proxy URL (default: uses built-in proxy)")
    parser.add_argument("--no-proxy", action="store_true",
                        help="Disable proxy")
    
    args = parser.parse_args()
    
    success = test_api_connection(
        provider=args.provider,
        api_key=args.api_key,
        model_name=args.model,
        proxy_url=args.proxy,
        use_proxy=not args.no_proxy
    )
    
    print("\n" + "=" * 60)
    if success:
        print(f"🎉 All tests passed! {args.provider.upper()} API is ready to use.")
    else:
        print(f"💥 Test failed. Please check your {args.provider.upper()} configuration.")
    print("=" * 60)
