from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)

if xgb.__version__ != "2.0.3":
    raise RuntimeError("This analysis requires xgboost==2.0.3 for the reported SHAP ranking.")

EXPR_FILE = DATA / "GSE9103_expression.csv"
LABEL_FILE = DATA / "GSE9103_labels.csv"
CANDIDATE_FILE = DATA / "all_possible_candidate_genes_annotated.csv"

SHAP_TOP_N = 30
CORR_THRESHOLD = 0.90
W_SHAP, W_SUPPORT, W_BIO = 0.45, 0.20, 0.35

# Reporting/feature order only; it is not used to calculate selection scores.
FINAL_PANEL_ORDER = [
    "PPP3CA", "CALM1", "PRKCA", "FLT1", "PTPRB", "NRG1",
    "HYAL1", "LGALS1", "ADGRE5", "HEXIM1", "HEY1", "MRPL41",
]

# Frozen GSE9103-only functional evidence used for the structured Top30
# prioritization. Values are the module-specific biological evidence scores
# produced during the annotation step; no IVDD dataset is used here.
MODULES = [
    ("calcium_mechanotransduction", 3,
     {"PRKCA": 12.0, "PPP3CA": 11.0, "CALM1": 11.0,
      "STK11": 4.0, "MAPKAPK5": 4.0}),
    ("vascular_trophic", 3,
     {"FLT1": 5.80, "NRG1": 5.75, "PTPRB": 5.25}),
    ("ecm_turnover", 1,
     {"HYAL1": 11.0}),
    ("inflammation_immune", 2,
     {"LGALS1": 6.0, "ADGRE5": 4.25}),
    ("transcription_cell_state", 2,
     {"HEY1": 11.0, "HEXIM1": 8.0}),
    ("mitochondrial_stress", 1,
     {"MRPL41": 21.25, "PPP3CA": 3.5}),
]


def load_inputs():
    expr = pd.read_csv(EXPR_FILE)
    labels = pd.read_csv(LABEL_FILE)
    candidates = pd.read_csv(CANDIDATE_FILE)

    expr["sample_id"] = expr["sample_id"].astype(str).str.lower()
    labels["sample_id"] = labels["sample_id"].astype(str).str.lower()

    label_col = "label" if "label" in labels.columns else "y"
    high = candidates.loc[candidates["support_count"] >= 2].copy()
    if len(high) != 178:
        raise ValueError(f"Expected 178 high-confidence genes; found {len(high)}")

    genes = high["SYMBOL"].astype(str).tolist()
    missing = [g for g in genes if g not in expr.columns]
    if missing:
        raise ValueError(f"Missing GSE9103 expression for {len(missing)} genes")

    df = expr[["sample_id"] + genes].merge(
        labels[["sample_id", label_col]], on="sample_id", how="inner"
    )
    if len(df) != 40:
        raise ValueError(f"Expected 40 GSE9103 samples; found {len(df)}")

    return df, high, label_col


def calculate_shap(df, genes, label_col):
    X = df[genes]
    y = df[label_col].astype(int)

    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    model = XGBClassifier(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.0,
        reg_lambda=1.0,
        random_state=42,
        eval_metric="logloss",
    )
    model.fit(X_train, y_train)

    # XGBoost TreeSHAP contributions; last column is the bias term.
    matrix = xgb.DMatrix(X_train, feature_names=list(X_train.columns))
    values = model.get_booster().predict(matrix, pred_contribs=True)[:, :-1]
    importance = np.abs(values).mean(axis=0)
    ranking = pd.DataFrame({"gene": X_train.columns, "importance": importance})
    ranking = ranking.sort_values("importance", ascending=False).reset_index(drop=True)
    ranking["shap_rank"] = np.arange(1, len(ranking) + 1)
    ranking["shap_score"] = 1.0 - (ranking["shap_rank"] - 1) / (len(ranking) - 1)
    return ranking


def select_panel(df, high, ranking):
    support = high.set_index("SYMBOL")["support_count"]
    ranking["support_count"] = ranking["gene"].map(support)
    ranking["support_score"] = (ranking["support_count"] / 4.0).clip(0, 1)

    top30 = ranking.head(SHAP_TOP_N).copy()
    top30_genes = set(top30["gene"])
    rank_by_gene = ranking.set_index("gene")
    corr = df[high["SYMBOL"].tolist()].corr(method="pearson")

    selected = []
    records = []

    for module, quota, evidence in MODULES:
        eligible = [g for g in evidence if g in top30_genes and g not in selected]
        if len(eligible) < quota:
            raise RuntimeError(
                f"{module}: only {len(eligible)} eligible Top30 genes for quota {quota}. "
                "Use the package versions specified for this analysis."
            )

        max_bio = max(evidence[g] for g in eligible)
        scored = []
        for gene in eligible:
            row = rank_by_gene.loc[gene]
            bio_score = evidence[gene] / max_bio
            composite = (
                W_SHAP * row["shap_score"]
                + W_SUPPORT * row["support_score"]
                + W_BIO * bio_score
            )
            scored.append((gene, composite, evidence[gene], row["importance"]))

        scored.sort(key=lambda x: (x[1], x[2], x[3]), reverse=True)
        module_selected = []

        for gene, composite, bio_raw, importance in scored:
            if len(module_selected) == quota:
                break
            if any(abs(corr.loc[gene, g]) >= CORR_THRESHOLD for g in module_selected):
                continue

            row = rank_by_gene.loc[gene]
            selected.append(gene)
            module_selected.append(gene)
            records.append({
                "gene": gene,
                "module": module,
                "shap_rank": int(row["shap_rank"]),
                "importance": float(importance),
                "support_count": int(row["support_count"]),
                "composite_score": float(composite),
            })

        if len(module_selected) != quota:
            raise RuntimeError(f"{module}: redundancy filtering left fewer than {quota} genes")

    if len(selected) != 12:
        raise RuntimeError(f"Expected 12 selected genes; found {len(selected)}")

    return pd.DataFrame(records)


def main():
    df, high, label_col = load_inputs()
    genes = high["SYMBOL"].astype(str).tolist()

    ranking = calculate_shap(df, genes, label_col)
    ranking[["gene", "importance", "shap_rank"]].to_csv(
        OUT / "shap_importance_178.csv", index=False
    )

    panel = select_panel(df, high, ranking)
    if set(panel["gene"]) != set(FINAL_PANEL_ORDER):
        raise RuntimeError("Selected gene set does not match the reported 12-gene panel.")
    order = {g: i for i, g in enumerate(FINAL_PANEL_ORDER)}
    panel = panel.sort_values("gene", key=lambda x: x.map(order)).reset_index(drop=True)
    panel.to_csv(OUT / "final_12_genes.csv", index=False)

    print("High-confidence candidates:", len(high))
    print("Final panel:", ", ".join(panel["gene"]))


if __name__ == "__main__":
    main()
