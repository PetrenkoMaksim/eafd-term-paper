#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PTLS_ENV="/workspace/venvs/ptls"

if [ ! -d "$PTLS_ENV" ]; then
    python3 -m venv "$PTLS_ENV"
fi

source "$PTLS_ENV/bin/activate"

python -m pip install --upgrade pip setuptools wheel
python -m pip install \
    pandas numpy scipy scikit-learn pyarrow tqdm \
    pyspark==3.5.1 \
    tensorboard tensorboardX \
    onnxruntime \
    hydra-core omegaconf \
    pytorch-lightning torchmetrics

python -m pip uninstall -y torch torchvision torchaudio triton || true
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

mkdir -p "$ROOT/external"
if [ ! -d "$ROOT/external/ptls-experiments/.git" ]; then
    git clone https://github.com/pytorch-lifestream/ptls-experiments.git "$ROOT/external/ptls-experiments"
fi

if [ ! -d "$ROOT/external/pytorch-lifestream/.git" ]; then
    git clone https://github.com/pytorch-lifestream/pytorch-lifestream.git "$ROOT/external/pytorch-lifestream"
fi

python -m pip install -e "$ROOT/external/pytorch-lifestream"

python - <<'PY'
import importlib.util
import pyspark
import sys
import torch

print("python:", sys.executable)
print("torch:", torch.__version__)
print("torch.cuda.is_available():", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
spec = importlib.util.find_spec("ptls")
print("ptls:", spec.origin if spec else "NOT FOUND")
print("pyspark:", pyspark.__version__)
PY
