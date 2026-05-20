#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[1]

PTLS_DATA = ROOT / "external" / "ptls-experiments" / "scenario_rosbank" / "data"
EAFD_ROOT = ROOT / "external" / "turbo-stats"
EAFD_DATA = EAFD_ROOT / "data" / "rosbank"
EAFD_EMB = EAFD_ROOT / "embeddings" / "rosbank"

RAW_TRAIN_PATH = PTLS_DATA / "train.csv"
EMBEDDINGS_PATH = PTLS_DATA / "mles_embeddings.pickle"
LEAKAGE_COLS = {"target_flag", "target_sum", "churn", "age", "gender"}
ID_COLS = ("cl_id", "client_id", "customer_id", "id")


def fail(message: str) -> None:
    raise SystemExit(f"ERROR: {message}")


def validate_inputs() -> None:
    if not EAFD_ROOT.exists():
        fail(f"turbo-stats repo not found: {EAFD_ROOT}")
    if not (EAFD_ROOT / "feature_agent" / "main.py").exists():
        fail(f"turbo-stats feature_agent/main.py not found under: {EAFD_ROOT}")
    if not RAW_TRAIN_PATH.exists():
        fail(f"raw Rosbank train.csv not found: {RAW_TRAIN_PATH}")
    if not EMBEDDINGS_PATH.exists():
        fail(f"PTLS embeddings pickle not found: {EMBEDDINGS_PATH}")


def as_dataframe(obj: object) -> pd.DataFrame:
    if isinstance(obj, pd.DataFrame):
        return obj.copy()
    if isinstance(obj, pd.Series):
        return obj.reset_index()
    if isinstance(obj, dict):
        try:
            return pd.DataFrame(obj)
        except ValueError:
            return pd.DataFrame(list(obj.items()), columns=["cl_id", "embedding"])
    fail(f"unsupported embeddings pickle type: {type(obj)!r}")


def detect_id_column(df: pd.DataFrame) -> str:
    for col in ID_COLS:
        if col in df.columns:
            return col
    fail(f"no client id column found in embeddings; tried {ID_COLS}; columns={df.columns.tolist()[:30]}")


def expand_vector_column(df: pd.DataFrame) -> pd.DataFrame:
    vector_cols: list[str] = []
    for col in df.columns:
        if col == "cl_id":
            continue
        sample = df[col].dropna().head(1)
        if sample.empty:
            continue
        value = sample.iloc[0]
        if isinstance(value, (list, tuple, np.ndarray)):
            vector_cols.append(col)

    if len(vector_cols) == 1:
        vector_col = vector_cols[0]
        values = df[vector_col].map(lambda x: np.asarray(x, dtype=np.float32))
        matrix = np.vstack(values.to_numpy())
        emb_values = pd.DataFrame(matrix, columns=[f"emb_{i}" for i in range(matrix.shape[1])])
        return pd.concat([df[["cl_id"]].reset_index(drop=True), emb_values], axis=1)

    if len(vector_cols) > 1:
        fail(f"multiple vector-like embedding columns found: {vector_cols}")

    numeric_cols = [c for c in df.columns if c != "cl_id" and pd.api.types.is_numeric_dtype(df[c])]
    if not numeric_cols:
        fail("no numeric embedding columns found after id normalization")
    return df[["cl_id", *numeric_cols]].copy()


def normalize_embeddings(path: Path, labeled_ids: set[int]) -> pd.DataFrame:
    emb = as_dataframe(pd.read_pickle(path))
    id_col = detect_id_column(emb)
    if id_col != "cl_id":
        emb = emb.rename(columns={id_col: "cl_id"})

    emb["cl_id"] = emb["cl_id"].astype(int)
    emb = expand_vector_column(emb)
    emb = emb.drop_duplicates("cl_id")
    emb = emb[emb["cl_id"].isin(labeled_ids)].sort_values("cl_id").reset_index(drop=True)
    return emb


def main() -> None:
    validate_inputs()
    EAFD_DATA.mkdir(parents=True, exist_ok=True)
    EAFD_EMB.mkdir(parents=True, exist_ok=True)

    print(f"Reading raw Rosbank transactions: {RAW_TRAIN_PATH}")
    raw = pd.read_csv(RAW_TRAIN_PATH)
    if "cl_id" not in raw.columns:
        fail("train.csv has no cl_id column")
    if "target_flag" not in raw.columns:
        fail("train.csv has no target_flag column")

    target = (
        raw[["cl_id", "target_flag"]]
        .drop_duplicates("cl_id")
        .sort_values("cl_id")
        .reset_index(drop=True)
    )
    target["cl_id"] = target["cl_id"].astype(int)
    if "target_flag" not in target.columns:
        fail("target_flag missing from target output")

    duplicate_labels = raw[["cl_id", "target_flag"]].drop_duplicates().duplicated("cl_id").sum()
    if duplicate_labels:
        fail(f"found {duplicate_labels} clients with inconsistent target_flag values")

    transactions = raw.drop(columns=[c for c in LEAKAGE_COLS if c in raw.columns], errors="ignore")
    transactions["cl_id"] = transactions["cl_id"].astype(int)
    leakage_left = sorted(LEAKAGE_COLS.intersection(transactions.columns))
    if leakage_left:
        fail(f"leakage columns remain in transactions: {leakage_left}")

    _, test_target = train_test_split(
        target,
        test_size=0.2,
        random_state=42,
        stratify=target["target_flag"],
    )
    test_ids = test_target[["cl_id"]].sort_values("cl_id").reset_index(drop=True)

    labeled_ids = set(target["cl_id"].astype(int))
    print(f"Reading PTLS embeddings: {EMBEDDINGS_PATH}")
    embeddings = normalize_embeddings(EMBEDDINGS_PATH, labeled_ids)

    missing_embeddings = sorted(labeled_ids - set(embeddings["cl_id"].astype(int)))
    test_not_in_target = sorted(set(test_ids["cl_id"].astype(int)) - labeled_ids)

    transactions.to_csv(EAFD_DATA / "transactions.csv", index=False)
    target.to_csv(EAFD_DATA / "target.csv", index=False)
    test_ids.to_csv(EAFD_DATA / "test_ids.csv", index=False)
    embeddings.to_pickle(EAFD_EMB / "coles.pickle")

    print("Saved EAFD files:")
    print(f"  {EAFD_DATA / 'transactions.csv'}")
    print(f"  {EAFD_DATA / 'target.csv'}")
    print(f"  {EAFD_DATA / 'test_ids.csv'}")
    print(f"  {EAFD_EMB / 'coles.pickle'}")
    print()
    print("Diagnostics:")
    print(f"  transactions shape: {transactions.shape}")
    print(f"  target shape: {target.shape}")
    print(f"  test_ids shape: {test_ids.shape}")
    print(f"  embeddings shape: {embeddings.shape}")
    print(f"  transaction first columns: {transactions.columns.tolist()[:20]}")
    print(f"  embedding first columns: {embeddings.columns.tolist()[:10]}")
    print(f"  missing embeddings count: {len(missing_embeddings)}")
    print(f"  test ids not in target count: {len(test_not_in_target)}")
    print(f"  leakage columns left in transactions: {leakage_left}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
