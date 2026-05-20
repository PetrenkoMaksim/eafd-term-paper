import pandas as pd
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, f1_score
from catboost import CatBoostClassifier

seed_everywhere = 42
COL_ID = 'customer_id'
COL_TARGET = 'gender'

def main():
    coles = pd.read_pickle('/home/jovyan/sakhno/turbo-stats/embeddings/gender/coles.pickle')
    coles[COL_ID] = coles[COL_ID].astype(int)
    test_ids = pd.read_csv('/home/jovyan/sakhno/turbo-stats/data/gender/test_ids.csv')
    test_ids[COL_ID] = test_ids[COL_ID].astype(int)

    labels = pd.read_csv('/home/jovyan/sakhno/turbo-stats/data/gender/target.csv')
    labels[COL_ID] = labels[COL_ID].astype(int)
    coles = coles.merge(labels, on=COL_ID)
    train = coles[~coles[COL_ID].isin(test_ids[COL_ID])]
    test = coles[coles[COL_ID].isin(test_ids[COL_ID])]
    

    final_model = CatBoostClassifier(logging_level='Silent', 
                                    depth=7,
                                    learning_rate=0.03,
                                    l2_leaf_reg=3,
                                    iterations=1000,
                                    random_state=seed_everywhere, 
                                    allow_writing_files=False)
    final_model.fit(train.drop(columns=[COL_ID, COL_TARGET]), train[COL_TARGET])
    test_pred = final_model.predict(test.drop(columns=[COL_ID, COL_TARGET]))
    test_proba = final_model.predict_proba(test.drop(columns=[COL_ID, COL_TARGET]))[:, 1]

    acc = balanced_accuracy_score(test[COL_TARGET], test_pred)
    roc_auc = roc_auc_score(test[COL_TARGET], test_proba)
    f1 = f1_score(test[COL_TARGET], test_pred, average='macro')
    
    print(f"Final Enhanced Model - Accuracy: {acc:.4f}, ROC-AUC: {roc_auc:.4f}, F1: {f1:.4f}")
if __name__ == "__main__":
    main()