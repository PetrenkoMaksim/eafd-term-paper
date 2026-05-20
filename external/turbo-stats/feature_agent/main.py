import multiprocessing as mp
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass

import pandas as pd
import numpy as np
import logging 
from utils import create_agent
from sklearn.metrics import roc_auc_score, accuracy_score
import torch
import hydra
from omegaconf import DictConfig
import os
from datetime import datetime

def seed_everything(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Set up logging (will be reconfigured in main with Hydra config)
logger = logging.getLogger("FeatureAgent")


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    # Set up logging from config
    log_level = getattr(logging, cfg.logging.level.upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(cfg.logging.log_file),
            logging.StreamHandler()
        ]
    )
    logger.setLevel(log_level)
    
    seed_everything(cfg.seed_everywhere)
    
    logger.info(f"Starting Feature Agent with {cfg.agent.type} backend")
    logger.info(f"Dataset: {cfg.dataset.col_id}, Target: {cfg.dataset.col_target}")
    
    # --- 1. Load data ---
    transactions = pd.read_csv(cfg.dataset.transactions_path)
    test_ids = pd.read_csv(cfg.dataset.test_ids_path)
    test_ids[cfg.dataset.col_id] = test_ids[cfg.dataset.col_id].astype(int)
    train_transactions = transactions[~transactions[cfg.dataset.col_id].isin(set(test_ids[cfg.dataset.col_id]))].reset_index(drop=True)
    test_transactions = transactions[transactions[cfg.dataset.col_id].isin(set(test_ids[cfg.dataset.col_id]))].reset_index(drop=True)
    
    # --- 2. Prepare labels and model ---
    labels = pd.read_csv(cfg.dataset.target_path)
    baseline_features = pd.read_pickle(cfg.dataset.baseline_features_path)

    baseline_features[cfg.dataset.col_id] = baseline_features[cfg.dataset.col_id].astype(int)
    baseline_features = baseline_features.merge(labels, on=cfg.dataset.col_id)

    train_baseline_features = baseline_features[~baseline_features[cfg.dataset.col_id].isin(set(test_ids[cfg.dataset.col_id]))].reset_index(drop=True)
    train_target = train_baseline_features[cfg.dataset.col_target]

    test_baseline_features = baseline_features[baseline_features[cfg.dataset.col_id].isin(set(test_ids[cfg.dataset.col_id]))].reset_index(drop=True)
    test_target = test_baseline_features[cfg.dataset.col_target].values

    train_baseline_features = train_baseline_features.drop(columns=[cfg.dataset.col_target])
    test_baseline_features = test_baseline_features.drop(columns=[cfg.dataset.col_target])

    now_str = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    if hasattr(cfg.training, 'output_dir') and cfg.training.output_dir:
        os.makedirs(cfg.training.output_dir, exist_ok=True)
        cfg.training.output_dir = os.path.join(cfg.training.output_dir, now_str)
    else:
        cfg.training.output_dir = None
    # Create agent based on Hydra configuration
    agent = create_agent(cfg, train_transactions, train_baseline_features, train_target)
    
    # Run the evolutionary search
    results = agent.evolutionary_search()
    logger.info(f"Evolutionary search completed with {len(results)} successful features")
    
    # Apply best features to data
    train_enhanced, test_enhanced = agent.apply_best_features_to_full_data(
        sequential_data_train=train_transactions,
        sequential_data_test=test_transactions,
        transformed_data_train=train_baseline_features,
        transformed_data_test=test_baseline_features
    )
    print(len(train_enhanced), len(test_enhanced))
    # Get feature importance
    # importance_df = agent.get_feature_importance()
    # print("Top feature importances:")
    # print(importance_df.head(10))
    
    # Evaluate the enhanced model
    final_model = hydra.utils.instantiate(cfg.model)
    final_model.fit(
        train_enhanced.drop(columns=[cfg.dataset.col_id]),
        train_target
    )
    final_metric = hydra.utils.get_method(cfg.dataset.main_metric)
    # Evaluate on the enhanced test set
    if final_metric is accuracy_score:
        test_pred = final_model.predict(test_enhanced.drop(columns=[cfg.dataset.col_id]))
        metric = accuracy_score(test_target, test_pred)
        print(f"Final Enhanced Model - Accuracy: {metric:.4f}")
    elif final_metric is roc_auc_score:
        test_pred = final_model.predict_proba(test_enhanced.drop(columns=[cfg.dataset.col_id]))[:, 1]
        metric = roc_auc_score(test_target, test_pred)
        print(f"Final Enhanced Model - ROC AUC: {metric:.4f}")

    results_to_save = {
        "Metric": [metric],
        "Metric_Name": [type(final_metric).__name__]
    }
    results_df = pd.DataFrame(results_to_save)
    if cfg.training.output_dir:
        results_df.to_csv(os.path.join(cfg.training.output_dir, "test_results.csv"), index=False)
    else:
        os.makedirs("results", exist_ok=True)
        results_df.to_csv(os.path.join("results", "test_results.csv"), index=False)
    
if __name__ == "__main__":
    main()
