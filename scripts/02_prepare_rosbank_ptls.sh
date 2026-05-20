#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PTLS_ENV="/workspace/venvs/ptls"
PTLS_REPO="$ROOT/external/ptls-experiments"

source "$PTLS_ENV/bin/activate"

mkdir -p "$ROOT/external"
if [ ! -d "$PTLS_REPO/.git" ]; then
    git clone https://github.com/pytorch-lifestream/ptls-experiments.git "$PTLS_REPO"
fi

cd "$PTLS_REPO/scenario_rosbank"
mkdir -p data logs /workspace/spark-tmp/yarn-local /workspace/tmp
chmod -R 777 /workspace/spark-tmp /workspace/tmp

if [ ! -f data/train.csv ] || [ ! -f data/test.csv ]; then
    sh bin/get-data.sh
else
    echo "data/train.csv and data/test.csv already exist, skipping download"
fi

decompress_if_gzip() {
    local path="$1"
    if [ -f "$path" ] && python - "$path" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
with path.open("rb") as f:
    magic = f.read(2)
raise SystemExit(0 if magic == b"\x1f\x8b" else 1)
PY
    then
        mv "$path" "$path.gz"
        gzip -d "$path.gz"
    fi
}

decompress_if_gzip data/train.csv
decompress_if_gzip data/test.csv

head -5 data/train.csv
head -5 data/test.csv

export LOCAL_DIRS=/workspace/spark-tmp/yarn-local
export SPARK_LOCAL_DIRS=/workspace/spark-tmp
export TMPDIR=/workspace/tmp
export SPARK_LOCAL_IP=127.0.0.1
export PYSPARK_PYTHON=/workspace/venvs/ptls/bin/python
export PYSPARK_DRIVER_PYTHON=/workspace/venvs/ptls/bin/python

spark-submit \
    --master "local[8]" \
    --conf spark.local.dir=/workspace/spark-tmp \
    --conf spark.driver.extraJavaOptions="-Djava.io.tmpdir=/workspace/tmp" \
    --conf spark.executor.extraJavaOptions="-Djava.io.tmpdir=/workspace/tmp" \
    --driver-memory 16G \
    make_dataset.py \
    --data_path data/ \
    --col_client_id "cl_id" \
    --cols_event_time "TRDATETIME" \
    --cols_category "mcc" "channel_type" "currency" "trx_category" \
    --cols_log_norm "amount" \
    --col_target "target_flag" "target_sum" \
    --output_train_path "data/train_trx.parquet" \
    --output_test_path "data/test_trx.parquet" \
    --output_test_ids_path "data/test_ids.csv" \
    2>&1 | tee logs/make_dataset_debug.log

test -d data/train_trx.parquet
test -d data/test_trx.parquet
test -f data/test_ids.csv

echo "Prepared PTLS Rosbank data:"
find data -maxdepth 2 -type f | head -30
