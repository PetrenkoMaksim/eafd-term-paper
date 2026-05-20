#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EAFD_ROOT="$ROOT/external/turbo-stats"
EAFD_ENV="/workspace/venvs/eafd"

source "$EAFD_ENV/bin/activate"
cd "$EAFD_ROOT"

RUN_NAME="rosbank_gptoss20b_iter5_seed42_$(date +%Y%m%d_%H%M%S)"
mkdir -p logs "outputs/$RUN_NAME"

python - <<'PY'
from pathlib import Path

path = Path("feature_agent/vllm_agent.py")
text = path.read_text()
text = text.replace("max_model_len=100000", "max_model_len=16384")
text = text.replace("max_model_len = 100000", "max_model_len = 16384")
path.write_text(text)
PY

export CUDA_VISIBLE_DEVICES=0
export HF_HOME=/workspace/hf_cache
export TRANSFORMERS_CACHE=/workspace/hf_cache
export VLLM_WORKER_MULTIPROC_METHOD=spawn

test -f data/rosbank/transactions.csv
test -f data/rosbank/target.csv
test -f data/rosbank/test_ids.csv
test -f embeddings/rosbank/coles.pickle

python feature_agent/main.py \
    dataset=rosbank \
    agent.type=vllm \
    agent.vllm.model_path=openai/gpt-oss-20b \
    agent.vllm.num_gpus=1 \
    agent.vllm.dtype=bfloat16 \
    agent.vllm.max_tokens=4096 \
    training.iterations=5 \
    training.n_tries=2 \
    training.reflection=true \
    training.context_window=20 \
    training.cols_budget=512 \
    training.output_dir="$PWD/outputs/$RUN_NAME" \
    training.seed=42 \
    model.iterations=1000 \
    seed_everywhere=42 \
    dataset.col_target=target_flag \
    dataset.transactions_path="$PWD/data/rosbank/transactions.csv" \
    dataset.test_ids_path="$PWD/data/rosbank/test_ids.csv" \
    dataset.target_path="$PWD/data/rosbank/target.csv" \
    dataset.baseline_features_path="$PWD/embeddings/rosbank/coles.pickle" \
    2>&1 | tee "logs/${RUN_NAME}.log"
