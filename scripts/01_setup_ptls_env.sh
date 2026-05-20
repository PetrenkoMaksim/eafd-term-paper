#!/usr/bin/env bash
set -euo pipefail

export PATH="$HOME/micromamba/bin:$PATH"

micromamba create -y -n ptls python=3.8
eval "$(micromamba shell hook --shell bash)"
micromamba activate ptls

python -m pip install --upgrade pip setuptools wheel

cd external/ptls-experiments

# По README у них основной setup через pipenv.
python -m pip install pipenv

# Можно попробовать их официальный путь:
pipenv sync --dev || true

# Если pipenv оказался болезненным, ставим минимально нужное руками.
python -m pip install \
    numpy pandas scipy scikit-learn matplotlib tqdm \
    pyspark tensorboard hydra-core hydra-optuna-sweeper \
    pytorch-lightning torchmetrics

python -m pip install "git+https://github.com/dllllb/pytorch-lifestream.git@v0.3.0"
python -m pip install "git+https://github.com/dllllb/ptls-validation.git"

python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
PY
