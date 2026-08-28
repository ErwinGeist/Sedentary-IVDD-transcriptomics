from pathlib import Path

import numpy as np
import pandas as pd

import xgboost as xgb
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix, f1_score,
    matthews_corrcoef, precision_score, recall_score, roc_auc_score
)
from xgboost import XGBClassifier


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

PANEL_FILE = RESULTS / "final_12_genes.csv"


def zscore(df, genes):
    out = df.copy()
    x = out[genes].apply(pd.to_numeric, errors="coerce")
    sd = x.std(axis=0).replace(0, np.nan)
    out[genes] = (x - x.mean(axis=0)) / sd
    return out


def load_gse9103(genes):
    expr = pd.read_csv(DATA / "GSE9103_expression.csv")
    meta = pd.read_csv(DATA / "GSE9103_labels.csv")

    expr["sample_id"] = expr["sample_id"].astype(str).str.lower()
    meta["sample_id"] = meta["sample_id"].astype(str).str.lower()
    if "y" in meta.columns:
        meta = meta.rename(columns={"y": "label"})

    df = expr[["sample_id"] + genes].merge(
        meta[["sample_id", "label"]], on="sample_id", how="inner"
    )
    df = zscore(df, genes)
    df["label"] = df["label"].astype(int)
    return df


def load_ivdd(dataset, genes):
    expr = pd.read_csv(DATA / f"{dataset}_expression.csv")
    meta = pd.read_csv(DATA / f"{dataset}_labels.csv")

    expr["sample_id"] = expr["sample_id"].astype(str).str.lower()
    meta["sample_id"] = meta["sample_id"].astype(str).str.lower()
    if "tissue_grade" in meta.columns and "grade" not in meta.columns:
        meta = meta.rename(columns={"tissue_grade": "grade"})

    df = expr[["sample_id"] + genes].merge(
        meta[["sample_id", "grade"]], on="sample_id", how="inner"
    )

    # Standardize the complete dataset first, including grade-3 samples.
    df = zscore(df, genes)

    df = df[df["grade"] != 3].copy()
    df["label"] = np.where(df["grade"] <= 2, 0, 1).astype(int)
    return df


def make_model():
    return XGBClassifier(
        n_estimators=600,
        max_depth=2,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.0,
        reg_lambda=1.0,
        eval_metric="logloss",
        random_state=42,
        n_jobs=4,
    )


def metrics(y, prob):
    pred = (prob >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "AUC": roc_auc_score(y, prob),
        "AUPR": average_precision_score(y, prob),
        "Accuracy": accuracy_score(y, pred),
        "Precision": precision_score(y, pred, zero_division=0),
        "Sensitivity": recall_score(y, pred, zero_division=0),
        "Specificity": tn / (tn + fp),
        "F1": f1_score(y, pred, zero_division=0),
        "MCC": matthews_corrcoef(y, pred),
    }


def main():
    if xgb.__version__ != "2.0.3":
        print(f"Warning: reported analysis used xgboost 2.0.3; detected {xgb.__version__}")

    genes = pd.read_csv(PANEL_FILE)["gene"].astype(str).tolist()

    train = load_gse9103(genes)
    d15227 = load_ivdd("GSE15227", genes)
    d23130 = load_ivdd("GSE23130", genes)

    model = make_model()
    model.fit(train[genes].to_numpy(), train["label"].to_numpy())

    rows = []
    pred_rows = []

    for name, df in [("GSE15227", d15227), ("GSE23130", d23130)]:
        y = df["label"].to_numpy()
        prob = model.predict_proba(df[genes].to_numpy())[:, 1]
        row = {"Dataset": name, "N": len(df), **metrics(y, prob)}
        rows.append(row)

        pred_rows.append(pd.DataFrame({
            "Dataset": name,
            "sample_id": df["sample_id"].values,
            "grade": df["grade"].values,
            "label": y,
            "probability": prob,
        }))

    result = pd.DataFrame(rows)
    result.to_csv(RESULTS / "cross_tissue_validation.csv", index=False)
    pd.concat(pred_rows, ignore_index=True).to_csv(
        RESULTS / "cross_tissue_predictions.csv", index=False
    )

    print(result[["Dataset", "AUC"]].to_string(index=False))


if __name__ == "__main__":
    main()
