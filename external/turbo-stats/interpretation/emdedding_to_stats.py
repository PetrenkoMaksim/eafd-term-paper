import os
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from catboost import CatBoostRegressor, Pool

EMBEDDING_PATH = 'embeddings/gender/gpt_transf.pickle'
STATS_PATH = 'statistics/turbo_stats_gender.parquet'
ID_COL = 'customer_id'


def train_all_targets(
    df,
    results_csv_path="interpretation/emb2stats_metrics.csv",
    test_size=0.20,
    random_state=42,
    gpu_devices="0",
    verbose=0,
):
    # 1. определить признаки и цели
    emb_cols = sorted([c for c in df.columns if c.startswith("emb_")])
    assert emb_cols, "Нет колонок emb_*."
    target_cols = [
        c for c in df.columns
        if c not in set(emb_cols + [ID_COL])
        and np.issubdtype(df[c].dtype, np.number)
    ]
    assert target_cols, "Нет числовых целей."

    X = df[emb_cols].values
    idx = np.arange(len(df))
    idx_tr, idx_te = train_test_split(idx, test_size=test_size, random_state=random_state)
    X_tr, X_te = X[idx_tr], X[idx_te]

    params = dict(
        loss_function="RMSE",
        learning_rate=0.05,
        depth=5,
        l2_leaf_reg=3.0,
        iterations=1000,
        od_type="Iter",
        od_wait=100,
        random_seed=random_state,
        task_type="GPU",
        devices=gpu_devices,
        allow_writing_files=False,
        verbose=verbose,
    )

    os.makedirs(os.path.dirname(results_csv_path) or ".", exist_ok=True)
    results = []

    print(f"Всего целей: {len(target_cols)}. Считаем метрики на заскейленных данных.\n")

    for i, tgt in enumerate(target_cols, 1):
        y = df[tgt].values.astype(float)
        y_tr, y_te = y[idx_tr], y[idx_te]

        if len(np.unique(y_tr)) <= 1:
            print(f"[{i}/{len(target_cols)}] {tgt}: пропуск (константная цель).")
            results.append({
                "target": tgt, "mae": np.nan, "rmse": np.nan, "r2": np.nan,
                "n_train": len(y_tr), "n_test": len(y_te),
                "fit_time_sec": 0.0, "note": "constant"
            })
            continue

        # масштабирование y
        scaler = StandardScaler()
        y_tr_s = scaler.fit_transform(y_tr.reshape(-1, 1)).ravel()
        y_te_s = scaler.transform(y_te.reshape(-1, 1)).ravel()

        start = time.time()
        try:
            model = CatBoostRegressor(**params)
            model.fit(Pool(X_tr, y_tr_s), eval_set=Pool(X_te, y_te_s), use_best_model=True)
        except Exception as e:
            print(f"   GPU недоступен, перехожу на CPU ({e})")
            model = CatBoostRegressor(**{**params, "task_type": "CPU"})
            model.fit(Pool(X_tr, y_tr_s), eval_set=Pool(X_te, y_te_s), use_best_model=True)
        fit_time = time.time() - start

        y_pred_s = model.predict(X_te)

        r2 = r2_score(y_te_s, y_pred_s)

        print(f"[{i}/{len(target_cols)}] {tgt:30s} | R²: {r2:.4f}")

        results.append({
            "target": tgt,
            "r2": r2,
        })

    res_df = pd.DataFrame(results).sort_values("r2")
    os.makedirs(os.path.dirname(results_csv_path) or ".", exist_ok=True)
    res_df.to_csv(results_csv_path, index=False)
    print(f"\nГотово. Метрики сохранены в: {results_csv_path}")
    return res_df

if __name__ == "__main__":
    embs = pd.read_pickle(EMBEDDING_PATH)
    stats = pd.read_parquet(STATS_PATH)

    embs[ID_COL] = embs[ID_COL].astype(int)
    stats[ID_COL] = stats[ID_COL].astype(int)

    df = embs.merge(stats, on=ID_COL)
    train_all_targets(df, results_csv_path="interpretation/gender/gpt_transf.csv")