from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


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
    meta = pd.read_csv(LABEL_FILE)
    panel = pd.read_csv(PANEL_FILE)

    expr["sample_id"] = expr["sample_id"].astype(str).str.lower()
    meta["sample_id"] = meta["sample_id"].astype(str).str.lower()

    if "y" in meta.columns and "label" not in meta.columns:
        meta = meta.rename(columns={"y": "label"})

    genes = panel["gene"].astype(str).tolist()
    missing = [g for g in genes if g not in expr.columns]
    if missing:
        raise ValueError(f"Missing genes: {missing}")

    df = expr[["sample_id"] + genes].merge(
        meta[["sample_id", "label"]],
        on="sample_id",
        how="inner",
    )

    x = df[genes].apply(pd.to_numeric, errors="raise")
    sd = x.std(axis=0, ddof=1).replace(0, np.nan)
    x = (x - x.mean(axis=0)) / sd

    return x.to_numpy(float), df["label"].astype(int).to_numpy(), df["sample_id"].to_numpy()


def get_models():
    return {
        "LR": LogisticRegression(
            penalty="l1",
            C=0.3,
            solver="liblinear",
            max_iter=20,
            random_state=42,
        ),
        "RF": RandomForestClassifier(
            n_estimators=3,
            random_state=42,
            n_jobs=-1,
        ),
        "KNN": KNeighborsClassifier(n_neighbors=30),
        "SVM": SVC(
            kernel="rbf",
            C=0.128,
            gamma="scale",
            probability=True,
            random_state=42,
        ),
        "XGB": xgb.XGBClassifier(
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
        ),
    }


def metrics(y_true, prob):
    pred = (prob >= THRESHOLD).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()

    return {
        "AUC": roc_auc_score(y_true, prob),
        "AUPR": average_precision_score(y_true, prob),
        "Accuracy": accuracy_score(y_true, pred),
        "Precision": precision_score(y_true, pred, zero_division=0),
        "Sensitivity": recall_score(y_true, pred, zero_division=0),
        "Specificity": tn / (tn + fp),
        "F1": f1_score(y_true, pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, pred),
    }


def repeated_fivefold(X, y):
    rows = []

    for seed in range(N_REPEATS):
        cv = StratifiedKFold(
            n_splits=N_FOLDS,
            shuffle=True,
            random_state=seed,
        )

        for name, model in get_models().items():
            oof = np.zeros(len(y), dtype=float)

            for train_idx, test_idx in cv.split(X, y):
                model.fit(X[train_idx], y[train_idx])
                oof[test_idx] = model.predict_proba(X[test_idx])[:, 1]

            rows.append({
                "Repeat": seed + 1,
                "Model": name,
                **metrics(y, oof),
            })

    detail = pd.DataFrame(rows)

    metric_cols = [
        "AUC", "AUPR", "Accuracy", "Precision",
        "Sensitivity", "Specificity", "F1", "MCC",
    ]

    summary_rows = []
    for model in get_models():
        d = detail[detail["Model"] == model]
        row = {"Model": model}
        for col in metric_cols:
            row[f"{col}_mean"] = d[col].mean()
            row[f"{col}_SD"] = d[col].std(ddof=0)
        summary_rows.append(row)

    return detail, pd.DataFrame(summary_rows)


def loocv(X, y, sample_ids):
    metric_rows = []
    prediction_rows = []

    loo = LeaveOneOut()

    for name, model in get_models().items():
        oof = np.zeros(len(y), dtype=float)

        for train_idx, test_idx in loo.split(X):
            model.fit(X[train_idx], y[train_idx])
            oof[test_idx[0]] = model.predict_proba(X[test_idx])[:, 1][0]

        metric_rows.append({
            "Model": name,
            **metrics(y, oof),
        })

        pred = (oof >= THRESHOLD).astype(int)
        prediction_rows.append(pd.DataFrame({
            "Model": name,
            "sample_id": sample_ids,
            "true_label": y,
            "pred_probability": oof,
            "pred_label": pred,
        }))

    return pd.DataFrame(metric_rows), pd.concat(prediction_rows, ignore_index=True)


def main():
    if xgb.__version__ != "2.0.3":
        print(
            f"Warning: xgboost {xgb.__version__} detected; "
            "the reported analysis used xgboost 2.0.3."
        )

    X, y, sample_ids = load_data()

    fivefold_detail, fivefold_summary = repeated_fivefold(X, y)
    loo_metrics, loo_predictions = loocv(X, y, sample_ids)

    fivefold_detail.to_csv(
        RESULTS / "model_comparison_10x5fold_detail.csv",
        index=False,
    )
    fivefold_summary.to_csv(
        RESULTS / "model_comparison_10x5fold_summary.csv",
        index=False,
    )
    loo_metrics.to_csv(
        RESULTS / "model_comparison_loocv.csv",
        index=False,
    )
    loo_predictions.to_csv(
        RESULTS / "model_comparison_loocv_predictions.csv",
        index=False,
    )

    print("\n10 x 5-fold:")
    print(
        fivefold_summary[
            ["Model", "AUC_mean", "AUC_SD", "Accuracy_mean", "F1_mean", "MCC_mean"]
        ].to_string(index=False)
    )

    print("\nLOOCV:")
    print(
        loo_metrics[
            ["Model", "AUC", "Accuracy", "F1", "MCC"]
        ].sort_values("AUC", ascending=False).to_string(index=False)
    )


if __name__ == "__main__":
    main()
