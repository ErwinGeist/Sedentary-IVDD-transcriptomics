from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import LeaveOneOut, StratifiedKFold
from xgboost import XGBClassifier


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

EXPR_FILE = DATA / "GSE9103_expression.csv"
LABEL_FILE = DATA / "GSE9103_labels.csv"
PANEL_FILE = RESULTS / "final_12_genes.csv"

N_REPEATS = 10
N_FOLDS = 5
THRESHOLD = 0.5


def load_data():
    expr = pd.read_csv(EXPR_FILE)
    labels = pd.read_csv(LABEL_FILE)
    panel = pd.read_csv(PANEL_FILE)

    expr["sample_id"] = expr["sample_id"].astype(str).str.lower()
    labels["sample_id"] = labels["sample_id"].astype(str).str.lower()

    label_col = "label" if "label" in labels.columns else "y"
    genes = panel["gene"].astype(str).tolist()

    missing = [g for g in genes if g not in expr.columns]
    if missing:
        raise ValueError(f"Missing genes in GSE9103 expression matrix: {missing}")

    df = expr[["sample_id"] + genes].merge(
        labels[["sample_id", label_col]],
        on="sample_id",
        how="inner",
    )

    if len(df) != 40:
        raise ValueError(f"Expected 40 GSE9103 samples; found {len(df)}")

    # Dataset-specific gene-wise z-score, matching the analysis pipeline.
    X = df[genes].apply(pd.to_numeric, errors="raise")
    sd = X.std(axis=0, ddof=1)
    if (sd == 0).any():
        raise ValueError("Zero-variance gene detected.")

    X = (X - X.mean(axis=0)) / sd
    y = df[label_col].astype(int).to_numpy()

    return X.to_numpy(), y, genes


def make_model():
    return XGBClassifier(
        n_estimators=300,
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


def calculate_metrics(y_true, y_prob):
    y_pred = (y_prob >= THRESHOLD).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return {
        "AUC": roc_auc_score(y_true, y_prob),
        "AUPR": average_precision_score(y_true, y_prob),
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Sensitivity": recall_score(y_true, y_pred, zero_division=0),
        "Specificity": tn / (tn + fp),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


def repeated_fivefold(X, y):
    rows = []

    for seed in range(N_REPEATS):
        cv = StratifiedKFold(
            n_splits=N_FOLDS,
            shuffle=True,
            random_state=seed,
        )

        oof_prob = np.zeros(len(y), dtype=float)

        for train_idx, test_idx in cv.split(X, y):
            model = make_model()
            model.fit(X[train_idx], y[train_idx])
            oof_prob[test_idx] = model.predict_proba(X[test_idx])[:, 1]

        row = {"Repeat": seed + 1, **calculate_metrics(y, oof_prob)}
        rows.append(row)

    detail = pd.DataFrame(rows)

    metric_cols = [
        "AUC", "AUPR", "Accuracy", "Precision",
        "Sensitivity", "Specificity", "F1", "MCC",
    ]

    summary = pd.DataFrame({
        "Metric": metric_cols,
        "Mean": [detail[c].mean() for c in metric_cols],
        "SD": [detail[c].std(ddof=0) for c in metric_cols],
    })

    return detail, summary


def leave_one_out(X, y):
    loo = LeaveOneOut()
    oof_prob = np.zeros(len(y), dtype=float)

    for train_idx, test_idx in loo.split(X):
        model = make_model()
        model.fit(X[train_idx], y[train_idx])
        oof_prob[test_idx] = model.predict_proba(X[test_idx])[:, 1]

    metrics = calculate_metrics(y, oof_prob)
    return pd.DataFrame([{"Validation": "LOOCV", **metrics}])


def main():
    if xgb.__version__ != "2.0.3":
        print(
            f"Warning: xgboost {xgb.__version__} detected; "
            "reported analysis used xgboost 2.0.3."
        )

    X, y, genes = load_data()

    fivefold_detail, fivefold_summary = repeated_fivefold(X, y)
    loocv_summary = leave_one_out(X, y)

    fivefold_detail.to_csv(
        RESULTS / "internal_5fold_repeats.csv",
        index=False,
    )
    fivefold_summary.to_csv(
        RESULTS / "internal_5fold_summary.csv",
        index=False,
    )
    loocv_summary.to_csv(
        RESULTS / "internal_loocv_summary.csv",
        index=False,
    )

    auc_5fold = float(
        fivefold_summary.loc[fivefold_summary["Metric"] == "AUC", "Mean"].iloc[0]
    )
    auc_loocv = float(loocv_summary["AUC"].iloc[0])

    print("Genes:", ", ".join(genes))
    print(f"Repeated 5-fold mean AUC: {auc_5fold:.4f} (~{auc_5fold:.2f})")
    print(f"LOOCV AUC: {auc_loocv:.4f} (~{auc_loocv:.2f})")


if __name__ == "__main__":
    main()
