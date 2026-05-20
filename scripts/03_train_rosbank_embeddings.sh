#!/usr/bin/env bash
set -euo pipefail

export PATH="$HOME/micromamba/bin:$PATH"
eval "$(micromamba shell hook --shell bash)"
micromamba activate ptls

cd external/ptls-experiments/scenario_rosbank

export CUDA_VISIBLE_DEVICES=0
mkdir -p models results

echo "Training CoLES/MeLES encoder..."
python -m ptls.pl_train_module \
    --config-dir conf \
    --config-name mles_params \
    trainer.max_epochs=60 \
    trainer.gpus=1 \
    model_path=models/mles_model.p

echo "Running embedding inference..."
python -m ptls.pl_inference \
    --config-dir conf \
    --config-name mles_params \
    model_path=models/mles_model.p \
    embed_file_name=mles_embeddings

echo "Done. Checking embeddings:"
ls -lh data/*embeddings* models/
