# EAFD Rosbank GPT-OSS-20B Configuration

This term-paper reproduction uses the original EAFD/turbo-stats code with PTLS-generated Rosbank embeddings.

## vLLM

Recommended settings for a single A100:

```yaml
model_path: openai/gpt-oss-20b
num_gpus: 1
dtype: bfloat16
max_model_len: 16384
max_tokens:
  smoke: 2048
  full: 4096
```

The original EAFD default `openai/gpt-oss-120b` / `num_gpus: 8` configuration is not used for this reproduction.

## Dataset Paths

All paths are relative to `external/turbo-stats`:

```text
data/rosbank/transactions.csv
data/rosbank/target.csv
data/rosbank/test_ids.csv
embeddings/rosbank/coles.pickle
```

Required Hydra overrides:

```text
dataset=rosbank
dataset.col_target=target_flag
dataset.transactions_path="$PWD/data/rosbank/transactions.csv"
dataset.test_ids_path="$PWD/data/rosbank/test_ids.csv"
dataset.target_path="$PWD/data/rosbank/target.csv"
dataset.baseline_features_path="$PWD/embeddings/rosbank/coles.pickle"
```

`transactions.csv` must not contain `target_flag`, `target_sum`, `churn`, `age`, or `gender`.

## Smoke Run

```text
agent.type=vllm
agent.vllm.model_path=openai/gpt-oss-20b
agent.vllm.num_gpus=1
agent.vllm.dtype=bfloat16
agent.vllm.max_tokens=2048
training.iterations=1
training.n_tries=1
training.context_window=5
training.cols_budget=256
model.iterations=100
```

## Full Run

```text
agent.type=vllm
agent.vllm.model_path=openai/gpt-oss-20b
agent.vllm.num_gpus=1
agent.vllm.dtype=bfloat16
agent.vllm.max_tokens=4096
training.iterations=5
training.n_tries=2
training.context_window=20
training.cols_budget=512
training.seed=42
seed_everywhere=42
model.iterations=1000
```
