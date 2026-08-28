from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)


def load_raw(dataset, genes, ivdd=False):
    expr = pd.read_csv(DATA / f"{dataset}_expression.csv")
    meta = pd.read_csv(DATA / f"{dataset}_labels.csv")

    expr["sample_id"] = expr["sample_id"].astype(str).str.lower()
    meta["sample_id"] = meta["sample_id"].astype(str).str.lower()

    if dataset == "GSE9103":
        if "y" in meta.columns:
            meta = meta.rename(columns={"y": "label"})
        return expr[["sample_id"] + genes].merge(
            meta[["sample_id", "label"]], on="sample_id", how="inner"
        )

    if "tissue_grade" in meta.columns and "grade" not in meta.columns:
        meta = meta.rename(columns={"tissue_grade": "grade"})
    return expr[["sample_id"] + genes].merge(
        meta[["sample_id", "grade"]], on="sample_id", how="inner"
    )


def zscore(df, genes):
    out = df.copy()
    x = out[genes].apply(pd.to_numeric, errors="coerce")
    sd = x.std(axis=0).replace(0, np.nan)
    out[genes] = (x - x.mean(axis=0)) / sd
    return out


def minmax(df, genes):
    out = df.copy()
    x = out[genes].apply(pd.to_numeric, errors="coerce")
    ranges = (x.max(axis=0) - x.min(axis=0)).replace(0, np.nan)
    out[genes] = (x - x.min(axis=0)) / ranges
    return out


def standard_subset(df):
    out = df[df["grade"] != 3].copy()
    out["label"] = np.where(out["grade"] <= 2, 0, 1).astype(int)
    return out


def model():
    return XGBClassifier(
        n_estimators=600, max_depth=2, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.0, reg_lambda=1.0,
        eval_metric="logloss", random_state=42, n_jobs=4
    )


def evaluate(method_name, transform, genes):
    train = transform(load_raw("GSE9103", genes), genes)
    d15227 = standard_subset(transform(load_raw("GSE15227", genes), genes))
    d23130 = standard_subset(transform(load_raw("GSE23130", genes), genes))

    train["label"] = train["label"].astype(int)

    clf = model()
    clf.fit(train[genes].to_numpy(float), train["label"].to_numpy(int))

    auc15227 = roc_auc_score(
        d15227["label"],
        clf.predict_proba(d15227[genes].to_numpy(float))[:, 1]
    )
    auc23130 = roc_auc_score(
        d23130["label"],
        clf.predict_proba(d23130[genes].to_numpy(float))[:, 1]
    )

    return {
        "Normalization": method_name,
        "GSE15227_AUC": auc15227,
        "GSE23130_AUC": auc23130,
        "Mean_validation_AUC": float(np.mean([auc15227, auc23130])),
    }


def main():
    genes = pd.read_csv(RESULTS / "final_12_genes.csv")["gene"].astype(str).tolist()

    result = pd.DataFrame([
        evaluate("Dataset-specific z-score", zscore, genes),
        evaluate("Dataset-specific Min-Max", minmax, genes),
    ])

    result.to_csv(RESULTS / "normalization_sensitivity.csv", index=False)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
