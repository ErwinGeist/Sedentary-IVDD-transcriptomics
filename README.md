# Sedentary status-associated transcriptomic signature and IVDD

Code supporting the analyses reported in:

**Cross-tissue transcriptomic analysis identifies a sedentary status-associated gene signature relevant to intervertebral disc degeneration**

## Data

Public GEO datasets used in this study:

- **GSE9103** — skeletal muscle, GPL570
- **GSE15227** — intervertebral disc, GPL1352
- **GSE23130** — intervertebral disc, GPL1352

Raw CEL files are available from NCBI GEO and are not redistributed in this repository.

The `data/` directory contains the processed gene-level expression matrices, sample labels, and the frozen candidate-gene annotation table used in the reported analysis:

```text
data/
├── GSE9103_expression.csv
├── GSE9103_labels.csv
├── GSE15227_expression.csv
├── GSE15227_labels.csv
├── GSE23130_expression.csv
├── GSE23130_labels.csv
└── all_possible_candidate_genes_annotated.csv
```

## Repository structure

```text
R/
└── 01_microarray_preprocessing.R

python/
├── 02_candidate_and_feature_selection.py
├── 03_internal_validation.py
├── 04_cross_tissue_validation.py
├── 05_phenotype_sensitivity.py
├── 06_panel_robustness.py
├── 07_random_panel_baseline.py
├── 08_normalization_sensitivity.py
├── 09_gene_expression_validation.py
├── 10_internal_model_comparison.py
└── 11_enrichment_and_candidate_pool.py
```

The `results/` directory is created automatically when the Python scripts are run.

## Installation

Install Python dependencies with:

```bash
pip install -r requirements.txt
```

R preprocessing requires the following packages:

- affy
- Biobase
- data.table
- R.utils
- AnnotationDbi
- hgu133plus2.db
- u133x3p.db
- hgu133plus2cdf
- u133x3pcdf

## Analysis workflow

The main reported analysis can be reproduced from the files already provided in `data/`.

Run:

```bash
python python/02_candidate_and_feature_selection.py
python python/03_internal_validation.py
python python/04_cross_tissue_validation.py
python python/05_phenotype_sensitivity.py
python python/06_panel_robustness.py
python python/07_random_panel_baseline.py
python python/08_normalization_sensitivity.py
python python/09_gene_expression_validation.py
python python/10_internal_model_comparison.py
```

`02_candidate_and_feature_selection.py` generates the SHAP ranking and the final 12-gene panel. Subsequent scripts use these generated files.

### Optional preprocessing from raw CEL files

To regenerate the processed expression matrices from GEO CEL files, place the files under:

```text
raw/
├── GSE9103/
├── GSE15227/
└── GSE23130/
```

and run:

```bash
Rscript R/01_microarray_preprocessing.R
```

The script performs RMA normalization, probe-to-gene annotation, and retains the maximum expression value when multiple probes map to the same gene.

### Optional enrichment reconstruction

`11_enrichment_and_candidate_pool.py` reconstructs the upstream differential-ranking, GO, and KEGG workflow.

```bash
python python/11_enrichment_and_candidate_pool.py
```

Because GO, KEGG/Enrichr, MyGene, and related online resources can change over time, the reported downstream analysis uses the frozen file:

```text
data/all_possible_candidate_genes_annotated.csv
```

This table contains the 449 candidate genes used in the study; applying `support_count >= 2` yields the 178 high-confidence candidates used for feature prioritization.

## Notes on validation

- Feature prioritization is based on GSE9103 and the frozen annotation evidence.
- GSE15227 and GSE23130 are not used for feature selection, hyperparameter selection, model optimization, or retraining.
- Each dataset is standardized independently.
- For the primary IVDD definition, grade 3 samples are excluded; grades <=2 are assigned to the low-degeneration group and grades >=4 to the high-degeneration group.
- Alternative grade-3 assignments are evaluated separately in the phenotype-sensitivity analysis.

## License

This repository is released under the MIT License. See `LICENSE`.
