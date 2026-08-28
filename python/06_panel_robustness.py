from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

SEED = 42


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
    df = zscore(df, genes)
    df = df[df["grade"] != 3].copy()
    df["label"] = np.where(df["grade"] <= 2, 0, 1).astype(int)
    return df


def internal_model():
    return XGBClassifier(
        n_estimators=300, max_depth=2, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.0, reg_lambda=1.0,
        eval_metric="logloss", random_state=42, n_jobs=4
    )


def transfer_model():
    return XGBClassifier(
        n_estimators=600, max_depth=2, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.0, reg_lambda=1.0,
        eval_metric="logloss", random_state=42, n_jobs=4
    )


def evaluate(panel, train, d15227, d23130):
    X = train[panel].to_numpy(float)
    y = train["label"].to_numpy(int)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof = np.zeros(len(y))
    fold_aucs = []

    for tr, te in cv.split(X, y):
        model = internal_model()
        model.fit(X[tr], y[tr])
        p = model.predict_proba(X[te])[:, 1]
        oof[te] = p
        fold_aucs.append(roc_auc_score(y[te], p))

    model = transfer_model()
    model.fit(X, y)

    auc15227 = roc_auc_score(
        d15227["label"],
        model.predict_proba(d15227[panel].to_numpy(float))[:, 1]
    )
    auc23130 = roc_auc_score(
        d23130["label"],
        model.predict_proba(d23130[panel].to_numpy(float))[:, 1]
    )

    return {
        "N_genes": len(panel),
        "GSE9103_OOF_AUC": roc_auc_score(y, oof),
        "GSE9103_mean_fold_AUC": float(np.mean(fold_aucs)),
        "GSE15227_AUC": auc15227,
        "GSE23130_AUC": auc23130,
        "Mean_validation_AUC": float(np.mean([auc15227, auc23130])),
        "Genes": ";".join(panel),
    }


def main():
    proposed = pd.read_csv(RESULTS / "final_12_genes.csv")["gene"].astype(str).tolist()
    shap = pd.read_csv(RESULTS / "shap_importance_178.csv").sort_values(
        "importance", ascending=False
    )

    all_genes = list(dict.fromkeys(proposed + shap["gene"].astype(str).tolist()))
    train = load_gse9103(all_genes)
    d15227 = load_ivdd("GSE15227", all_genes)
    d23130 = load_ivdd("GSE23130", all_genes)

    common = set(train.columns) & set(d15227.columns) & set(d23130.columns)
    shap_common = shap[shap["gene"].isin(common)].copy()

    panels = {
        "Proposed_12": proposed,
        "SHAP_Top6": shap_common.head(6)["gene"].tolist(),
        "SHAP_Top12": shap_common.head(12)["gene"].tolist(),
        "SHAP_Top24": shap_common.head(24)["gene"].tolist(),
    }

    rows = []
    definitions = []

    for name, panel in panels.items():
        rows.append({"Panel": name, **evaluate(panel, train, d15227, d23130)})
        for rank, gene in enumerate(panel, start=1):
            definitions.append({"Panel": name, "Rank": rank, "Gene": gene})

    pd.DataFrame(rows).to_csv(
        RESULTS / "panel_robustness_summary.csv", index=False
    )
    pd.DataFrame(definitions).to_csv(
        RESULTS / "panel_definitions.csv", index=False
    )


if __name__ == "__main__":
    main()
