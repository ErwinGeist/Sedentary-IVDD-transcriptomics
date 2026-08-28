from pathlib import Path
import gzip
import shutil
import urllib.request

import numpy as np
import pandas as pd
import gseapy as gp
from mygene import MyGeneInfo
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests
from goatools.obo_parser import GODag
from goatools.associations import read_ncbi_gene2go
from goatools.go_enrichment import GOEnrichmentStudy


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

EXPR_FILE = DATA / "GSE9103_expression.csv"
LABEL_FILE = DATA / "GSE9103_labels.csv"
FROZEN_CANDIDATE_FILE = DATA / "all_possible_candidate_genes_annotated.csv"

GO_DIR = RESULTS / "_go_data"
GO_DIR.mkdir(exist_ok=True)

OBO_FILE = GO_DIR / "go-basic.obo"
GENE2GO_FILE = GO_DIR / "gene2go"

GO_OBO_URL = "https://current.geneontology.org/ontology/go-basic.obo"
GENE2GO_URL = "https://ftp.ncbi.nlm.nih.gov/gene/DATA/gene2go.gz"

DISC_KEYWORDS = [
    "extracellular", "matrix", "collagen", "proteoglycan",
    "cartilage", "chondro", "connective tissue", "skeletal system",
    "intervertebral", "disc", "nucleus pulposus", "annulus fibrosus",
    "cell adhesion", "adhesion", "inflamm", "immune", "cytokine",
    "chemokine", "oxidative stress", "response to stress",
    "reactive oxygen", "apoptotic", "apoptosis", "cell death",
    "aging", "senescence", "hypoxia", "angiogenesis", "wound healing",
    "tissue remodeling", "development",
]

KEGG_TARGET_TERMS = [
    "MAPK signaling pathway",
    "PI3K-Akt signaling pathway",
    "Apoptosis",
    "Focal adhesion",
    "Cellular senescence",
    "ErbB signaling pathway",
    "Proteasome",
]


def load_gse9103():
    expr = pd.read_csv(EXPR_FILE)
    meta = pd.read_csv(LABEL_FILE)

    expr["sample_id"] = expr["sample_id"].astype(str).str.lower()
    meta["sample_id"] = meta["sample_id"].astype(str).str.lower()

    if "y" in meta.columns and "label" not in meta.columns:
        meta = meta.rename(columns={"y": "label"})

    return expr.merge(
        meta[["sample_id", "label"]],
        on="sample_id",
        how="inner",
    )


def differential_ranking(df):
    gene_cols = [c for c in df.columns if c not in {"sample_id", "label"}]
    rows = []

    for gene in gene_cols:
        y0 = pd.to_numeric(
            df.loc[df["label"] == 0, gene],
            errors="coerce",
        ).dropna()
        y1 = pd.to_numeric(
            df.loc[df["label"] == 1, gene],
            errors="coerce",
        ).dropna()

        t_stat, p_value = ttest_ind(
            y1,
            y0,
            equal_var=False,
            nan_policy="omit",
        )

        log2fc = y1.mean() - y0.mean()

        rows.append({
            "gene": gene,
            "mean_y0": y0.mean(),
            "mean_y1": y1.mean(),
            "log2FC(y1-y0)": log2fc,
            "abs_log2FC": abs(log2fc),
            "t": t_stat,
            "p_t": p_value,
        })

    out = pd.DataFrame(rows)
    out["q_t(BH)"] = multipletests(
        out["p_t"].fillna(1.0).to_numpy(),
        method="fdr_bh",
    )[1]

    out = out.sort_values(
        ["q_t(BH)", "p_t", "abs_log2FC"],
        ascending=[True, True, False],
    ).reset_index(drop=True)

    return out


def download_go_files():
    if not OBO_FILE.exists():
        urllib.request.urlretrieve(GO_OBO_URL, OBO_FILE)

    if not GENE2GO_FILE.exists():
        gz_path = GO_DIR / "gene2go.gz"
        urllib.request.urlretrieve(GENE2GO_URL, gz_path)
        with gzip.open(gz_path, "rb") as src, open(GENE2GO_FILE, "wb") as dst:
            shutil.copyfileobj(src, dst)
        gz_path.unlink(missing_ok=True)


def map_symbols_to_entrez(symbols):
    mg = MyGeneInfo()
    res = mg.querymany(
        symbols,
        scopes="symbol",
        fields="entrezgene,symbol",
        species="human",
        as_dataframe=False,
        verbose=False,
    )

    rows = []
    for r in res:
        if r.get("notfound", False):
            continue
        if r.get("entrezgene") is None:
            continue
        rows.append({
            "query_symbol": r.get("query"),
            "mapped_symbol": r.get("symbol"),
            "entrezgene": int(r.get("entrezgene")),
        })

    return (
        pd.DataFrame(rows)
        .drop_duplicates(subset=["query_symbol"], keep="first")
    )


def go_candidates(stats, top_n):
    top = stats.head(top_n).copy()
    symbols = top["gene"].dropna().astype(str).drop_duplicates().tolist()
    mapping = map_symbols_to_entrez(symbols)

    mapped = top.merge(
        mapping,
        left_on="gene",
        right_on="query_symbol",
        how="inner",
    )

    obodag = GODag(str(OBO_FILE))
    geneid2gos = read_ncbi_gene2go(
        str(GENE2GO_FILE),
        taxids=[9606],
    )

    goea = GOEnrichmentStudy(
        list(geneid2gos.keys()),
        geneid2gos,
        obodag,
        propagate_counts=False,
        alpha=0.05,
        methods=["fdr_bh"],
    )

    results = goea.run_study(
        mapped["entrezgene"].drop_duplicates().astype(int).tolist()
    )

    rows = []
    result_by_go = {}

    for r in results:
        result_by_go[r.GO] = r
        if not hasattr(r, "p_fdr_bh") or pd.isna(r.p_fdr_bh):
            continue

        rows.append({
            "GO": r.GO,
            "term_name": r.name,
            "namespace": r.NS,
            "p_uncorrected": r.p_uncorrected,
            "p_fdr_bh": r.p_fdr_bh,
            "study_count": r.study_count,
            "study_n": r.study_n,
            "pop_count": r.pop_count,
            "pop_n": r.pop_n,
        })

    go_df = pd.DataFrame(rows).sort_values(
        ["p_fdr_bh", "study_count"],
        ascending=[True, False],
    )

    go_df.to_csv(
        RESULTS / f"go_top{top_n}_all.csv",
        index=False,
    )

    go_df[go_df["p_fdr_bh"] < 0.05].to_csv(
        RESULTS / f"go_top{top_n}_significant.csv",
        index=False,
    )

    pattern = "|".join(DISC_KEYWORDS)
    disc = go_df[
        go_df["term_name"].str.contains(
            pattern,
            case=False,
            na=False,
            regex=True,
        )
    ].copy()

    focus = disc[
        (disc["p_fdr_bh"] < 0.20)
        | (disc["p_uncorrected"] < 0.01)
    ].copy()

    focus.to_csv(
        RESULTS / f"go_top{top_n}_disc_focus.csv",
        index=False,
    )

    candidate_ids = set()
    for go_id in focus["GO"]:
        candidate_ids.update(result_by_go[go_id].study_items)

    id_to_symbol = (
        mapping.dropna(subset=["mapped_symbol"])
        .drop_duplicates("entrezgene")
        .set_index("entrezgene")["mapped_symbol"]
        .to_dict()
    )

    genes = {
        id_to_symbol[g]
        for g in candidate_ids
        if g in id_to_symbol and id_to_symbol[g]
    }

    return genes


def kegg_core(go_gene_set, top_n):
    enr = gp.enrichr(
        gene_list=sorted(go_gene_set),
        gene_sets="KEGG_2021_Human",
        outdir=None,
        cutoff=1.0,
    )

    res = enr.results.copy()
    res = res.sort_values("Adjusted P-value", ascending=True)

    res.to_csv(
        RESULTS / f"kegg_top{top_n}_all.csv",
        index=False,
    )

    sig = res[res["Adjusted P-value"] < 0.05].copy()
    sig.to_csv(
        RESULTS / f"kegg_top{top_n}_significant.csv",
        index=False,
    )

    selected = sig[sig["Term"].isin(KEGG_TARGET_TERMS)]

    genes = set()
    for value in selected["Genes"].dropna():
        genes.update(
            g.strip()
            for g in str(value).split(";")
            if g.strip()
        )

    return genes


def build_live_candidate_table(go1000, go3000, kegg1000, kegg3000):
    sources = {
        "Top1000_candidate": go1000,
        "Top3000_candidate": go3000,
        "Top1000_KEGG_core": kegg1000,
        "Top3000_KEGG_core": kegg3000,
    }

    all_genes = sorted(set().union(*sources.values()))
    rows = []

    for gene in all_genes:
        row = {"SYMBOL": gene}
        hits = []

        for name, genes in sources.items():
            row[name] = int(gene in genes)
            if gene in genes:
                hits.append(name)

        row["support_count"] = len(hits)
        row["support_sources"] = ";".join(hits)
        rows.append(row)

    return (
        pd.DataFrame(rows)
        .sort_values(["support_count", "SYMBOL"], ascending=[False, True])
        .reset_index(drop=True)
    )


def main():
    df = load_gse9103()
    stats = differential_ranking(df)

    stats.to_csv(
        RESULTS / "GSE9103_differential_ranking.csv",
        index=False,
    )
    stats.head(1000).to_csv(
        RESULTS / "GSE9103_top1000_genes.csv",
        index=False,
    )
    stats.head(3000).to_csv(
        RESULTS / "GSE9103_top3000_genes.csv",
        index=False,
    )

    download_go_files()

    go1000 = go_candidates(stats, 1000)
    go3000 = go_candidates(stats, 3000)

    kegg1000 = kegg_core(go1000, 1000)
    kegg3000 = kegg_core(go3000, 3000)

    live_table = build_live_candidate_table(
        go1000,
        go3000,
        kegg1000,
        kegg3000,
    )
    live_table.to_csv(
        RESULTS / "candidate_pool_live_rebuild.csv",
        index=False,
    )

    # Frozen evidence table used in the reported analysis.
    candidate = pd.read_csv(FROZEN_CANDIDATE_FILE)
    required = {
        "SYMBOL",
        "Top1000_candidate",
        "Top3000_candidate",
        "Top1000_KEGG_core",
        "Top3000_KEGG_core",
        "support_count",
        "support_sources",
    }
    if not required.issubset(candidate.columns):
        raise ValueError("Candidate annotation table has unexpected columns.")

    candidate = candidate.drop_duplicates("SYMBOL").copy()
    high_conf = candidate[candidate["support_count"] >= 2].copy()

    if len(candidate) != 449:
        raise ValueError(f"Expected 449 candidate genes; found {len(candidate)}")
    if len(high_conf) != 178:
        raise ValueError(f"Expected 178 high-confidence genes; found {len(high_conf)}")

    candidate.to_csv(
        RESULTS / "candidate_pool_449.csv",
        index=False,
    )
    high_conf.to_csv(
        RESULTS / "high_confidence_178_genes.csv",
        index=False,
    )

    print("Candidate pool:", len(candidate))
    print("High-confidence candidates:", len(high_conf))


if __name__ == "__main__":
    main()
