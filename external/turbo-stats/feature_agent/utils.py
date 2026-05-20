import torch
import os
import logging
import hydra

logger = logging.getLogger("FeatureAgent")


def create_agent(cfg, train_transactions, train_baseline_features, train_target):
    """Create the appropriate agent based on Hydra configuration."""

    common_params = {
        'sequential_data': train_transactions,
        'transformed_data': train_baseline_features,
        'target': train_target,
        'model': hydra.utils.instantiate(cfg.model),
        'reflection': cfg.training.reflection,
        'client_id_col': cfg.dataset.col_id,
        'eval_metric': hydra.utils.get_method(cfg.dataset.main_metric),
        'iterations': cfg.training.iterations,
        'test_size': cfg.training.test_size,
        'temperature': cfg.training.temperature,
        'random_state': cfg.seed_everywhere,
        'cols_budget': cfg.training.cols_budget,
        'n_tries': cfg.training.n_tries,
        'context_window': cfg.training.context_window,
        'output_dir': cfg.training.output_dir,
        'mode': cfg.training.mode
    }
    
    agent_type = cfg.agent.type
    
    if agent_type == 'vllm':
        from vllm_agent import VLLMFeatureAgent
        
        vllm_cfg = cfg.agent.vllm
        dtype_map = {'bfloat16': torch.bfloat16, 'float16': torch.float16, 'float32': torch.float32}
        dtype = dtype_map.get(vllm_cfg.dtype, torch.bfloat16)
        
        agent = VLLMFeatureAgent(
            model_path=vllm_cfg.model_path,
            num_gpus=vllm_cfg.num_gpus,
            dtype=dtype,
            max_tokens=vllm_cfg.max_tokens,
            **common_params
        )
        logger.info(f"Created VLLMFeatureAgent with model: {vllm_cfg.model_path}, GPUs: {vllm_cfg.num_gpus}")
        
    elif agent_type == 'openai':
        from openai_agent import OpenAIFeatureAgent
        
        openai_cfg = cfg.agent.openai
        api_key = openai_cfg.api_key or os.getenv('OPENAI_API_KEY')
        
        agent = OpenAIFeatureAgent(
            model_name=openai_cfg.model,
            api_key=api_key,
            base_url=openai_cfg.base_url,
            proxy_url=cfg.agent.proxy.url if cfg.agent.proxy.use_proxy else None,
            use_proxy=cfg.agent.proxy.use_proxy,
            max_tokens=openai_cfg.max_tokens,
            **common_params
        )
        proxy_status = "disabled" if not cfg.agent.proxy.use_proxy else (cfg.agent.proxy.url or "default")
        logger.info(f"Created OpenAIFeatureAgent with model: {openai_cfg.model}, proxy: {proxy_status}")
    
    elif agent_type == 'openrouter':
        from openai_agent import OpenRouterFeatureAgent
        
        openrouter_cfg = cfg.agent.openrouter
        api_key = openrouter_cfg.api_key or os.getenv('OPENROUTER_API_KEY')
        
        agent = OpenRouterFeatureAgent(
            model_name=openrouter_cfg.model,
            api_key=api_key,
            proxy_url=cfg.agent.proxy.url if cfg.agent.proxy.use_proxy else None,
            use_proxy=cfg.agent.proxy.use_proxy,
            max_tokens=openrouter_cfg.max_tokens,
            **common_params
        )
        proxy_status = "disabled" if not cfg.agent.proxy.use_proxy else (cfg.agent.proxy.url or "default")
        logger.info(f"Created OpenRouterFeatureAgent with model: {openrouter_cfg.model}, proxy: {proxy_status}")
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")
    
    return agent