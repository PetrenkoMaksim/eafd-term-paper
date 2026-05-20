#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PTLS_ENV="/workspace/venvs/ptls"
MODE="${1:-all}"
SCENARIO="$ROOT/external/ptls-experiments/scenario_rosbank"

source "$PTLS_ENV/bin/activate"
cd "$SCENARIO"

mkdir -p models results logs

patch_configs() {
    local cfg="conf/mles_params.yaml"
    if [ -f "$cfg" ] && [ ! -f "$cfg.bak" ]; then
        cp "$cfg" "$cfg.bak"
    fi

    python - <<'PY'
from pathlib import Path

keys_to_remove = {
    "gpus",
    "auto_select_gpus",
    "weights_summary",
    "progress_bar_refresh_rate",
    "checkpoint_callback",
    "resume_from_checkpoint",
}

for path in Path("conf").glob("*.yaml"):
    lines = path.read_text().splitlines()
    new_lines = []
    skip_nested = False
    skip_indent = 0

    for line in lines:
        stripped = line.lstrip()
        indent = len(line) - len(stripped)

        if skip_nested and stripped and indent > skip_indent:
            continue
        skip_nested = False

        key = stripped.split(":", 1)[0] if ":" in stripped else None
        if key in keys_to_remove:
            skip_nested = line.rstrip().endswith(":")
            skip_indent = indent
            continue
        new_lines.append(line)

    text = "\n".join(new_lines) + "\n"
    if "_target_: torch.load" in text or "_target_: torch.serialization.load" in text:
        out = []
        inserted = False
        for line in text.splitlines():
            out.append(line)
            if ("_target_: torch.load" in line or "_target_: torch.serialization.load" in line) and not inserted:
                indent = len(line) - len(line.lstrip())
                out.append(" " * indent + "weights_only: false")
                inserted = True
        text = "\n".join(out) + "\n"
    path.write_text(text)
PY
}

cuda_diagnostics() {
    nvidia-smi || true
    python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY
}

run_smoke() {
    python -m ptls.pl_train_module \
        --config-dir conf \
        --config-name mles_params \
        +trainer.accelerator=gpu \
        +trainer.devices=1 \
        trainer.max_epochs=1 \
        model_path=models/mles_model_smoke.p \
        2>&1 | tee logs/train_mles_smoke.log
}

run_full() {
    python -m ptls.pl_train_module \
        --config-dir conf \
        --config-name mles_params \
        +trainer.accelerator=gpu \
        +trainer.devices=1 \
        trainer.max_epochs=60 \
        model_path=models/mles_model.p \
        2>&1 | tee logs/train_mles_full.log
}

run_inference() {
    set +e
    python -m ptls.pl_inference \
        --config-dir conf \
        --config-name mles_params \
        model_path=models/mles_model.p \
        embed_file_name=mles_embeddings \
        inference.seq_encoder.weights_only=false \
        2>&1 | tee logs/inference_mles_embeddings.log
    local status=${PIPESTATUS[0]}
    set -e

    if [ "$status" -ne 0 ]; then
        echo "Retrying inference with +inference.seq_encoder.weights_only=false"
        python -m ptls.pl_inference \
            --config-dir conf \
            --config-name mles_params \
            model_path=models/mles_model.p \
            embed_file_name=mles_embeddings \
            +inference.seq_encoder.weights_only=false \
            2>&1 | tee logs/inference_mles_embeddings.log
    fi

    test -f data/mles_embeddings.pickle
    python - <<'PY'
import pandas as pd

df = pd.read_pickle("data/mles_embeddings.pickle")
print("mles_embeddings shape:", getattr(df, "shape", None))
print(df.head())
PY
}

patch_configs
cuda_diagnostics

case "$MODE" in
    smoke)
        run_smoke
        ;;
    full)
        run_full
        ;;
    inference)
        run_inference
        ;;
    all)
        run_smoke
        run_full
        run_inference
        ;;
    *)
        echo "Usage: $0 {smoke|full|inference|all}" >&2
        exit 2
        ;;
esac
