from pathlib import Path
import shutil
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[1]

PTLS_DATA = ROOT / "external" / "ptls-experiments" / "scenario_rosbank" / "data"
EAFD_ROOT = ROOT / "external" / "turbo-stats"

EAFD_DATA = EAFD_ROOT / "data" / "rosbank"
EAFD_EMB = EAFD_ROOT / "embeddings" / "rosbank"

EAFD_DATA.mkdir(parents=True, exist_ok=True)
EAFD_EMB.mkdir(parents=True, exist_ok=True)


def normalize_embeddings(path: Path) -> pd.DataFrame:
    obj = pd.read_pickle(path)

    if isinstance(obj, pd.DataFrame):
        emb = obj.copy()
    elif isinstance(obj, dict):
        emb = pd.DataFrame(obj)
    else:
        raise TypeError(f"Unknown embeddings pickle type: {type(obj)}")

    # Try to detect id column
    possible_id_cols = ["cl_id", "client_id", "customer_id", "id"]
    id_cols = [c for c in possible_id_cols if c in emb.columns]
    if not id_cols:
        raise ValueError(f"Cannot find id column in embeddings. Columns: {emb.columns.tolist()[:20]}")

    id_col = id_cols[0]
    if id_col != "cl_id":
        emb = emb.rename(columns={id_col: "cl_id"})

    # Common case: one column contains vector/list/np.ndarray
    vector_cols = []
    for c in emb.columns:
        if c == "cl_id":
            continue
        first_non_null = emb[c].dropna().head(1)
        if len(first_non_null) == 0:
            continue
        v = first_non_null.iloc[0]
        if isinstance(v, (list, tuple, np.ndarray)):
            vector_cols.append(c)

    if len(vector_cols) == 1:
        vc = vector_cols[0]
        arr = np.vstack(emb[vc].to_numpy())
        emb_values = pd.DataFrame(arr, columns=[f"emb_{i}" for i in range(arr.shape[1])])
        emb = pd.concat([emb[["cl_id"]].reset_index(drop=True), emb_values], axis=1)

    # Otherwise assume embeddings are already expanded into numeric columns.
    emb["cl_id"] = emb["cl_id"].astype(int)

    # Remove duplicate ids if any
    emb = emb.drop_duplicates("cl_id")

    return emb


def main():
    raw_train_path = PTLS_DATA / "train.csv"
    embeddings_path = PTLS_DATA / "mles_embeddings.pickle"

    if not raw_train_path.exists():
        raise FileNotFoundError(raw_train_path)

    if not embeddings_path.exists():
        raise FileNotFoundError(embeddings_path)

    print("Reading raw Rosbank train.csv...")
    trx = pd.read_csv(raw_train_path)

    required_cols = {"cl_id", "target_flag"}
    missing = required_cols - set(trx.columns)
    if missing:
        raise ValueError(f"Missing required columns in train.csv: {missing}")

    # One label per client
    target = (
        trx[["cl_id", "target_flag"]]
        .drop_duplicates("cl_id")
        .sort_values("cl_id")
        .reset_index(drop=True)
    )
    target["cl_id"] = target["cl_id"].astype(int)

    # Avoid leakage: transactions.csv must NOT contain labels.
    leakage_cols = ["target_flag", "target_sum", "churn", "age_bin", "gender"]
    transactions = trx.drop(columns=[c for c in leakage_cols if c in trx.columns], errors="ignore")
    transactions["cl_id"] = transactions["cl_id"].astype(int)

    # Use PTLS test_ids if available; otherwise create our own stratified holdout.
    ptls_test_ids_path = PTLS_DATA / "test_ids.csv"
    if ptls_test_ids_path.exists():
        test_ids = pd.read_csv(ptls_test_ids_path)
        if "cl_id" not in test_ids.columns:
            raise ValueError(f"test_ids.csv has no cl_id column. Columns: {test_ids.columns.tolist()}")
        test_ids = test_ids[["cl_id"]].drop_duplicates()
        test_ids["cl_id"] = test_ids["cl_id"].astype(int)
        test_ids = test_ids[test_ids["cl_id"].isin(set(target["cl_id"]))]
        print(f"Using PTLS test_ids: {len(test_ids)} clients")
    else:
        _, test_ids_full = train_test_split(
            target,
            test_size=0.2,
            random_state=42,
            stratify=target["target_flag"],
        )
        test_ids = test_ids_full[["cl_id"]]
        print(f"Created stratified test_ids: {len(test_ids)} clients")

    print("Reading and normalizing embeddings...")
    emb = normalize_embeddings(embeddings_path)

    # Keep only labeled clients
    emb = emb[emb["cl_id"].isin(set(target["cl_id"]))].reset_index(drop=True)

    missing_emb = set(target["cl_id"]) - set(emb["cl_id"])
    if missing_emb:
        print(f"WARNING: {len(missing_emb)} labeled clients have no embeddings")

    # Save EAFD-ready files
    transactions.to_csv(EAFD_DATA / "transactions.csv", index=False)
    target.to_csv(EAFD_DATA / "target.csv", index=False)
    test_ids.to_csv(EAFD_DATA / "test_ids.csv", index=False)
    emb.to_pickle(EAFD_EMB / "coles.pickle")

    print("Saved:")
    print(EAFD_DATA / "transactions.csv")
    print(EAFD_DATA / "target.csv")
    print(EAFD_DATA / "test_ids.csv")
    print(EAFD_EMB / "coles.pickle")

    print("\nShapes:")
    print("transactions:", transactions.shape)
    print("target:", target.shape)
    print("test_ids:", test_ids.shape)
    print("embeddings:", emb.shape)

    print("\nLeakage check:")
    print("Forbidden columns in transactions:",
          [c for c in leakage_cols if c in transactions.columns])


if __name__ == "__main__":
    main()
