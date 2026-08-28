from pathlib import Path

import numpy as np
import pandas as pd

from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

N_RANDOM = 1000
SEED = 42
N_JOBS = 6


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

    available = [g for g in genes if g in expr.columns]
    df = expr[["sample_id"] + available].merge(
        meta[["sample_id", "label"]], on="sample_id", how="inner"
    )
    df = zscore(df, available)
    df["label"] = df["label"].astype(int)
    return df, available


def load_ivdd(dataset, genes):
    expr = pd.read_csv(DATA / f"{dataset}_expression.csv")
    meta = pd.read_csv(DATA / f"{dataset}_labels.csv")
    expr["sample_id"] = expr["sample_id"].astype(str).str.lower()
    meta["sample_id"] = meta["sample_id"].astype(str).str.lower()
    if "tissue_grade" in meta.columns and "grade" not in meta.columns:
        meta = meta.rename(columns={"tissue_grade": "grade"})

    available = [g for g in genes if g in expr.columns]
    df = expr[["sample_id"] + available].merge(
        meta[["sample_id", "grade"]], on="sample_id", how="inner"
    )
    df = zscore(df, available)
    df = df[df["grade"] != 3].copy()
    df["label"] = np.where(df["grade"] <= 2, 0, 1).astype(int)
    return df, available


def internal_model():
    return XGBClassifier(
        n_estimators=300, max_depth=2, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.0, reg_lambda=1.0,
        eval_metric="logloss", random_state=42, n_jobs=1
    )


def transfer_model():
    return XGBClassifier(
        n_estimators=600, max_depth=2, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.0, reg_lambda=1.0,
        eval_metric="logloss", random_state=42, n_jobs=1
    )


def evaluate(panel, train, d15227, d23130, splits):
    X = train[panel].to_numpy(float)
    y = train["label"].to_numpy(int)

    oof = np.zeros(len(y))
    for tr, te in splits:
        model = internal_model()
        model.fit(X[tr], y[tr])
        oof[te] = model.predict_proba(X[te])[:, 1]

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
        "GSE9103_OOF_AUC": roc_auc_score(y, oof),
        "GSE15227_AUC": auc15227,
        "GSE23130_AUC": auc23130,
        "Mean_validation_AUC": float(np.mean([auc15227, auc23130])),
    }


def main():
    candidate_df = pd.read_csv(DATA / "all_possible_candidate_genes_annotated.csv")
    candidates = candidate_df["SYMBOL"].dropna().astype(str).drop_duplicates().tolist()
    if len(candidates) != 449:
        raise ValueError(f"Expected 449 candidate genes; found {len(candidates)}")

    train, g1 = load_gse9103(candidates)
    d15227, g2 = load_ivdd("GSE15227", candidates)
    d23130, g3 = load_ivdd("GSE23130", candidates)

    common = [g for g in candidates if g in set(g1) & set(g2) & set(g3)]

    valid = []
    for g in common:
        if all(
            np.isfinite(df[g].to_numpy(float)).all()
            for df in [train, d15227, d23130]
        ):
            valid.append(g)

    if len(valid) != 442:
        raise ValueError(f"Expected 442 valid genes for random panels; found {len(valid)}")

    proposed = pd.read_csv(RESULTS / "final_12_genes.csv")["gene"].astype(str).tolist()

    y = train["label"].to_numpy(int)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    splits = list(cv.split(np.zeros(len(y)), y))

    observed = evaluate(proposed, train, d15227, d23130, splits)

    rng = np.random.default_rng(SEED)
    panels = []
    seen = set()

    while len(panels) < N_RANDOM:
        panel = tuple(sorted(rng.choice(valid, size=12, replace=False).tolist()))
        if panel not in seen:
            seen.add(panel)
            panels.append(list(panel))

    def run_one(i, panel):
        return {
            "Random_ID": i + 1,
            "Genes": ";".join(panel),
            **evaluate(panel, train, d15227, d23130, splits),
        }

    random_rows = Parallel(n_jobs=N_JOBS, backend="loky")(
        delayed(run_one)(i, panel) for i, panel in enumerate(panels)
    )
    random_df = pd.DataFrame(random_rows)

    rv = random_df["Mean_validation_AUC"].to_numpy(float)
    empirical_p = (np.sum(rv >= observed["Mean_validation_AUC"]) + 1) / (len(rv) + 1)
    percentile = np.mean(rv < observed["Mean_validation_AUC"]) * 100

    summary = pd.DataFrame([{
        "Panel": "Proposed_12",
        **observed,
        "Random_background_size": len(valid),
        "Percentile": percentile,
        "Empirical_P": empirical_p,
    }, {
        "Panel": "Random_12_x1000_mean",
        "GSE9103_OOF_AUC": random_df["GSE9103_OOF_AUC"].mean(),
        "GSE15227_AUC": random_df["GSE15227_AUC"].mean(),
        "GSE23130_AUC": random_df["GSE23130_AUC"].mean(),
        "Mean_validation_AUC": random_df["Mean_validation_AUC"].mean(),
        "Random_background_size": len(valid),
        "Percentile": np.nan,
        "Empirical_P": np.nan,
    }])

    random_df.to_csv(RESULTS / "random_panel_results.csv", index=False)
    summary.to_csv(RESULTS / "random_panel_summary.csv", index=False)


if __name__ == "__main__":
    main()
