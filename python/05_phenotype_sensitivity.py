from pathlib import Path

import numpy as np
import pandas as pd

import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix, f1_score,
    matthews_corrcoef, precision_score, recall_score, roc_auc_score
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


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


def load_ivdd_full(dataset, genes):
    expr = pd.read_csv(DATA / f"{dataset}_expression.csv")
    meta = pd.read_csv(DATA / f"{dataset}_labels.csv")
    expr["sample_id"] = expr["sample_id"].astype(str).str.lower()
    meta["sample_id"] = meta["sample_id"].astype(str).str.lower()
    if "tissue_grade" in meta.columns and "grade" not in meta.columns:
        meta = meta.rename(columns={"tissue_grade": "grade"})

    df = expr[["sample_id"] + genes].merge(
        meta[["sample_id", "grade"]], on="sample_id", how="inner"
    )
    return zscore(df, genes)


def build_variants(df, dataset):
    standard = df[df["grade"] != 3].copy()
    standard["label"] = np.where(standard["grade"] <= 2, 0, 1)

    grade3_low = df.copy()
    grade3_low["label"] = np.where(grade3_low["grade"] <= 3, 0, 1)

    grade3_high = df.copy()
    grade3_high["label"] = np.where(grade3_high["grade"] <= 2, 0, 1)

    return {
        f"{dataset}_standard": standard,
        f"{dataset}_grade3_low": grade3_low,
        f"{dataset}_grade3_high": grade3_high,
    }


def get_models():
    return {
        "LR": LogisticRegression(
            penalty="l1", C=0.2, solver="liblinear",
            max_iter=20, random_state=42
        ),
        "RF": RandomForestClassifier(
            n_estimators=2, random_state=42, n_jobs=-1
        ),
        "KNN": KNeighborsClassifier(n_neighbors=35),
        "SVM": SVC(
            kernel="rbf", C=0.115, gamma="scale",
            probability=True, random_state=42
        ),
        "XGB": xgb.XGBClassifier(
            n_estimators=600, max_depth=2, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.0, reg_lambda=1.0,
            eval_metric="logloss", random_state=42, n_jobs=4
        ),
    }


def metric_row(y, prob):
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
    genes = pd.read_csv(PANEL_FILE)["gene"].astype(str).tolist()
    train = load_gse9103(genes)

    val_sets = {}
    for dataset in ["GSE15227", "GSE23130"]:
        val_sets.update(build_variants(load_ivdd_full(dataset, genes), dataset))

    rows = []
    pred_rows = []

    for model_name, model in get_models().items():
        model.fit(train[genes].to_numpy(), train["label"].to_numpy())

        for subset_name, df in val_sets.items():
            y = df["label"].astype(int).to_numpy()
            prob = model.predict_proba(df[genes].to_numpy())[:, 1]

            rows.append({
                "Dataset": subset_name,
                "Model": model_name,
                "N": len(df),
                "N_positive": int((y == 1).sum()),
                "N_negative": int((y == 0).sum()),
                **metric_row(y, prob),
            })

            pred_rows.append(pd.DataFrame({
                "Dataset": subset_name,
                "Model": model_name,
                "sample_id": df["sample_id"].values,
                "grade": df["grade"].values,
                "true_label": y,
                "pred_prob": prob,
            }))

    pd.DataFrame(rows).to_csv(
        RESULTS / "phenotype_sensitivity_metrics.csv", index=False
    )
    pd.concat(pred_rows, ignore_index=True).to_csv(
        RESULTS / "phenotype_sensitivity_predictions.csv", index=False
    )


if __name__ == "__main__":
    main()
