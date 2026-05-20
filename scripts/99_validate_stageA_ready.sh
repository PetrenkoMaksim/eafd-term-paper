#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

test -f external/turbo-stats/data/rosbank/transactions.csv
test -f external/turbo-stats/data/rosbank/target.csv
test -f external/turbo-stats/data/rosbank/test_ids.csv
test -f external/turbo-stats/embeddings/rosbank/coles.pickle

python - <<'PY'
from pathlib import Path
import sys

import pandas as pd

root = Path.cwd()
data = root / "external" / "turbo-stats" / "data" / "rosbank"
emb_path = root / "external" / "turbo-stats" / "embeddings" / "rosbank" / "coles.pickle"
leakage = {"target_flag", "target_sum", "churn", "age", "gender"}

transactions = pd.read_csv(data / "transactions.csv")
target = pd.read_csv(data / "target.csv")
test_ids = pd.read_csv(data / "test_ids.csv")
emb = pd.read_pickle(emb_path)

errors = []
if "cl_id" not in transactions.columns:
    errors.append("transactions.csv has no cl_id")
if "cl_id" not in target.columns or "target_flag" not in target.columns:
    errors.append("target.csv must contain cl_id,target_flag")
if "cl_id" not in test_ids.columns:
    errors.append("test_ids.csv has no cl_id")
if "cl_id" not in emb.columns:
    errors.append("coles.pickle has no cl_id")

leakage_left = sorted(leakage.intersection(transactions.columns))
if leakage_left:
    errors.append(f"transactions.csv contains leakage columns: {leakage_left}")

target_ids = set(target["cl_id"].astype(int))
test_missing = set(test_ids["cl_id"].astype(int)) - target_ids
emb_missing = target_ids - set(emb["cl_id"].astype(int))
if test_missing:
    errors.append(f"{len(test_missing)} test ids are not in target")
if emb_missing:
    errors.append(f"{len(emb_missing)} labeled clients have no embedding")

print("Stage A EAFD-ready diagnostics:")
print("  transactions:", transactions.shape)
print("  target:", target.shape)
print("  test_ids:", test_ids.shape)
print("  embeddings:", emb.shape)
print("  leakage columns:", leakage_left)
print("  missing embeddings:", len(emb_missing))
print("  test ids not in target:", len(test_missing))

if errors:
    for error in errors:
        print(f"ERROR: {error}", file=sys.stderr)
    raise SystemExit(1)
PY
