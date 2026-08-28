from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)


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

    x = df[genes].apply(pd.to_numeric, errors="coerce")
    sd = x.std(axis=0).replace(0, np.nan)
    df[genes] = (x - x.mean(axis=0)) / sd

    df = df[df["grade"] != 3].copy()
    df["label"] = np.where(df["grade"] <= 2, 0, 1).astype(int)
    return df


def delta_table(df, genes, dataset):
    rows = []
    for gene in genes:
        low = df.loc[df["label"] == 0, gene].astype(float)
        high = df.loc[df["label"] == 1, gene].astype(float)
        delta = high.mean() - low.mean()
        rows.append({
            "Gene": gene,
            f"{dataset}_Delta_z": delta,
            f"{dataset}_Direction": "Up" if delta > 0 else "Down" if delta < 0 else "No change",
        })
    return pd.DataFrame(rows)


def main():
    genes = pd.read_csv(RESULTS / "final_12_genes.csv")["gene"].astype(str).tolist()

    d15227 = load_ivdd("GSE15227", genes)
    d23130 = load_ivdd("GSE23130", genes)

    t1 = delta_table(d15227, genes, "GSE15227")
    t2 = delta_table(d23130, genes, "GSE23130")
    out = t1.merge(t2, on="Gene")

    out["Consistency"] = np.where(
        out["GSE15227_Direction"] == out["GSE23130_Direction"],
        "Consistent",
        "Opposite",
    )

    out.to_csv(RESULTS / "gene_expression_validation.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
