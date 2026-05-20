# AutoFeature: LLM-Powered Feature Engineering

Automated feature generation using Large Language Models with evolutionary optimization.

## Directory Overview

```
autofeature/
├── config.py                 # Configuration, imports, and utilities
├── feature_factory.py        # Manual feature engineering (FeatureFactoryRosbank)
├── evaluation.py             # Model evaluation and comparison utilities
├── feature_agent.py          # Core FeatureAgent with LLM integration
├── vllm_agent.py            # Optimized VLLMFeatureAgent for faster inference
├── code_execution.py         # Safe LLM code execution engine
├── prompts.py               # LLM prompt templates and configurations
└── main.py                  # Example usage and pipeline orchestration
```

## Quick Start

```python
from vllm_agent import VLLMFeatureAgent

# Initialize agent
agent = VLLMFeatureAgent(
    model_path="your-llm-model",
    sequential_data=transactions,
    transformed_data=features,
    target=labels,
    model=CatBoostClassifier(),
    iterations=10
)

# Generate features
results = agent.evolutionary_search()

# Apply best features
train_enhanced, test_enhanced = agent.apply_best_features_to_full_data()
```

## Core Components

### 🏗️ FeatureFactoryRosbank (`feature_factory.py`)
- Manual feature engineering from transaction data
- Aggregations, TF-IDF features, statistical metrics
- Income/expense categorization

### 🤖 FeatureAgent (`feature_agent.py`) 
- LLM-powered feature generation
- Evolutionary search with memory
- Safe code execution and validation

### ⚡ VLLMFeatureAgent (`vllm_agent.py`)
- High-performance version with vLLM
- Batch generation and execution
- GPU-optimized inference

### 🔧 Support Modules
- **`code_execution.py`**: Safe LLM code execution engine
- **`evaluation.py`**: Model evaluation and comparison
- **`prompts.py`**: LLM prompt templates
- **`config.py`**: Dependencies and configuration

## Usage

### 1. Manual Features
```python
factory = FeatureFactoryRosbank()
factory.fit(transactions)
features = factory.transform(new_data)
```

### 2. Automated Features
```python
agent = FeatureAgent(llm, transactions, features, target, model)
best_features = agent.evolutionary_search()
```

### 3. Evaluation
```python
evaluator = ModelEvaluator()
metrics = evaluator.compare_features(original, enhanced, target)
```

## Configuration

```python
# Core parameters
iterations=10,      # Evolutionary iterations
n_tries=3,          # Attempts per iteration  
test_size=0.3,      # Validation split
context_window=5,   # Memory size
```

## Data Requirements

- **Sequential Data**: Transaction records with client IDs
- **Transformed Data**: Aggregated features (from FeatureFactory)
- **Target**: Labels aligned with client IDs

## Installation

```bash
pip install pandas numpy scikit-learn catboost transformers torch
pip install vllm  # Optional: for GPU acceleration
```

## Architecture

```
Raw Data → FeatureFactory → Base Features → FeatureAgent → Enhanced Features
                                     │
                     LLM + Evolutionary Search
```
