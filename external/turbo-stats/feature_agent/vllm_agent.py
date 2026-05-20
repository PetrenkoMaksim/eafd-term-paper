import os
import pandas as pd
import torch
from typing import List, Literal
from vllm import LLM, SamplingParams

from feature_agent import FeatureAgent
from prompts import PREVIOUS_ERROR_TEMPLATE, REFLECTION_PROMPT_TEMPLATE, REFLECTION_INTERPRETATION_PROMPT_TEMPLATE, GENERATION_REFLECTED_PROMPT_TEMPLATE, GENERATION_PROMPT_TEMPLATE


class VLLMFeatureAgent(FeatureAgent):
    """FeatureAgent with vLLM integration for faster inference."""
    
    def __init__(self, 
                model_path: str, 
                num_gpus: int = 4,
                temperature: float = 0.3,
                max_tokens: int = 2000,
                dtype: torch.dtype = torch.bfloat16,
                **kwargs):
        # Initialize vLLM first
        os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
        self.debug_iteration = 0
        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=num_gpus,  # Adjust based on your GPU setup
            task='generate',
            gpu_memory_utilization=0.90,
            max_model_len=100000,
            dtype=dtype,
            trust_remote_code=True,
            enforce_eager=False, 
            disable_custom_all_reduce=True  
        )
        
        self.sampling_params = SamplingParams(
            temperature=temperature,
            top_p=0.95,
            max_tokens=max_tokens,
            stop_token_ids=[self.llm.get_tokenizer().eos_token_id]
        )
        
        # Remove llm from kwargs since we're handling it separately
        kwargs.pop('llm', None)
        super().__init__(llm=self.llm, **kwargs)
        
        # Override the parent's llm reference
        self._vllm_model = self.llm
        self._vllm_sampling_params = self.sampling_params
        
    def _generate_feature_code(self) -> str:
        """Generate feature code using vLLM with enhanced prompting."""

        def generate_code_or_prompt(prompt: str, mode: Literal["reflection", "generation"]):
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ]
            tokenizer = self.llm.get_tokenizer()

            enhanced_prompt = tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            outputs = self._vllm_model.generate(
                [enhanced_prompt], 
                self._vllm_sampling_params
            )
                
            generated_text = outputs[0].outputs[0].text
            if mode == "reflection":
                return generated_text
            else:
                code = self._extract_code_from_response(generated_text)
                if not code:
                    self.logger.warning("No code extracted")
                    return None
                return code

        try:
            if self.previous_error:
                # print(self.previous_error)
                self.debug_iteration += 1
                print(f"Debug iteration-----------------------------------------: {self.debug_iteration}")
                enhanced_prompt = self._build_enhanced_prompt(None, None, "debugging")
                return generate_code_or_prompt(enhanced_prompt, "generation")
            elif self.reflection:
                previous_features_section = self._build_previous_features()
                reflection_prompt = self._build_enhanced_prompt(None, previous_features_section, "reflection")
                generation_prompt = generate_code_or_prompt(reflection_prompt, "reflection")
                print(generation_prompt)
                enhanced_prompt = self._build_enhanced_prompt(generation_prompt, None, "generation")
                return generate_code_or_prompt(enhanced_prompt, "generation")
            else:
                previous_features_section = self._build_previous_features()
                enhanced_prompt = self._build_enhanced_prompt(None, previous_features_section, "generation")
                return generate_code_or_prompt(enhanced_prompt, "generation")
        except Exception as e:
            # self.logger.warning(f"Generation failed: {e}")
            return None
    
    def _build_enhanced_prompt(self, 
                                reflection_rules: str, 
                                previous_features_section: str, 
                                step: Literal["reflection", "generation", "debugging"]) -> str:
        """Build enhanced prompt with very specific instructions."""
        if step == "reflection":
            # Format feature importances if available
            feature_importances_section = ""
            interpretation_importance_section = ""
            if hasattr(self, 'current_features_importances_df') and self.current_features_importances_df is not None:
                top_features = self.current_features_importances_df.head(50)
                if not top_features.empty:
                    if self.mode == 'interpretation':
                        interpretation_importance_section = f"""
                        {top_features.to_string(index=False)}
                    """
                    else:
                        feature_importances_section = f"""
                        {top_features.to_string(index=False)}
                    """
            
            # Use interpretation prompt for interpretation mode
            if self.mode == 'interpretation':
                return REFLECTION_INTERPRETATION_PROMPT_TEMPLATE.format(
                    # previous_features_section=previous_features_section,
                    interpretation_importance_section=interpretation_importance_section
                )
            else:
                return REFLECTION_PROMPT_TEMPLATE.format(
                    previous_features_section=previous_features_section,
                    feature_importances_section=feature_importances_section
                )
        elif step == "debugging":

            return PREVIOUS_ERROR_TEMPLATE.format(
                error_message=self.previous_error,
                client_id_col=self.client_id_col
            )
        elif step == "generation":
            if self.reflection:
                return GENERATION_REFLECTED_PROMPT_TEMPLATE.format(
                    reflection_rules=reflection_rules,
                    client_id_col=self.client_id_col
                )
            else:
                return GENERATION_PROMPT_TEMPLATE.format(
                    previous_features_section=previous_features_section,
                    client_id_col=self.client_id_col
                )
        else:
            raise ValueError(f"Invalid step: {step}")