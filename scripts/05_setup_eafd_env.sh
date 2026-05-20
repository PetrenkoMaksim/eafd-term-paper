#!/usr/bin/env bash
set -euo pipefail

EAFD_ENV="/workspace/venvs/eafd"

if [ ! -d "$EAFD_ENV" ]; then
    python3 -m venv "$EAFD_ENV"
fi

source "$EAFD_ENV/bin/activate"

python -m pip install --upgrade pip setuptools wheel
python -m pip install \
    pandas numpy scipy scikit-learn catboost matplotlib tqdm \
    hydra-core omegaconf openai httpx pyarrow ipykernel
python -m pip install -U vllm --torch-backend=auto

python - <<'PY'
import importlib.metadata
import torch

print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
try:
    print("vllm:", importlib.metadata.version("vllm"))
except importlib.metadata.PackageNotFoundError:
    print("vllm: NOT FOUND")
PY
