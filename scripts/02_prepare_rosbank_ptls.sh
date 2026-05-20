#!/usr/bin/env bash
set -euo pipefail

export PATH="$HOME/micromamba/bin:$PATH"
eval "$(micromamba shell hook --shell bash)"
micromamba activate ptls

cd external/ptls-experiments/scenario_rosbank

mkdir -p results

# 1. Download raw Rosbank data
if [ ! -f data/train.csv ]; then
    sh bin/get-data.sh
else
    echo "data/train.csv already exists, skipping download"
fi

# 2. Convert raw transactions into PTLS parquet format
if [ ! -f data/train_trx.parquet/_SUCCESS ] && [ ! -d data/train_trx.parquet ]; then
    sh bin/make-datasets-spark.sh
else
    echo "train_trx.parquet already exists, skipping Spark preprocessing"
fi

echo "Prepared files:"
find data -maxdepth 2 -type f | head -30
