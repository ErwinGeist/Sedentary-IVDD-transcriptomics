#!/usr/bin/env Rscript

# GSE9103, GSE15227 and GSE23130 microarray preprocessing
# CEL/CEL.gz -> RMA -> probe-to-gene mapping -> max probe per gene
#
# Expected input:
#   raw/GSE9103/*.CEL[.gz]
#   raw/GSE15227/*.CEL[.gz]
#   raw/GSE23130/*.CEL[.gz]
#
# Output:
#   data/GSE9103_expression.csv
#   data/GSE15227_expression.csv
#   data/GSE23130_expression.csv
#
# Usage:
#   Rscript R/01_microarray_preprocessing.R [raw_root] [output_dir]

required <- c(
  "affy", "Biobase", "data.table", "R.utils", "AnnotationDbi",
  "hgu133plus2.db", "u133x3p.db", "hgu133plus2cdf", "u133x3pcdf"
)
missing <- required[!vapply(required, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing)) {
  stop("Missing package(s): ", paste(missing, collapse = ", "), call. = FALSE)
}

args <- commandArgs(trailingOnly = TRUE)
raw_root <- if (length(args) >= 1) args[1] else "raw"
out_dir  <- if (length(args) >= 2) args[2] else "data"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

get_cel_files <- function(path) {
  if (!dir.exists(path)) stop("Directory not found: ", path, call. = FALSE)

  cel <- list.files(path, pattern = "\\.(CEL|cel)$", full.names = TRUE)
  gz  <- list.files(path, pattern = "\\.(CEL|cel)\\.gz$", full.names = TRUE)

  if (!length(cel) && length(gz)) {
    invisible(lapply(gz, R.utils::gunzip, overwrite = TRUE, remove = FALSE))
    cel <- list.files(path, pattern = "\\.(CEL|cel)$", full.names = TRUE)
  }

  if (!length(cel)) stop("No CEL files found in: ", path, call. = FALSE)
  sort(cel)
}

collapse_to_gene <- function(expr_sample_probe, annotation_db) {
  map <- AnnotationDbi::select(
    annotation_db,
    keys = colnames(expr_sample_probe),
    keytype = "PROBEID",
    columns = "SYMBOL"
  )
  map <- unique(map[!is.na(map$SYMBOL) & map$SYMBOL != "", c("PROBEID", "SYMBOL")])
  data.table::setDT(map)

  x <- data.table::as.data.table(expr_sample_probe, keep.rownames = "sample_id")
  x <- data.table::melt(
    x,
    id.vars = "sample_id",
    variable.name = "PROBEID",
    value.name = "expr"
  )
  x <- merge(x, map, by = "PROBEID", all = FALSE)

  # When multiple probes map to one gene, retain the maximum expression value.
  x <- x[, .(expr = max(expr, na.rm = TRUE)), by = .(sample_id, SYMBOL)]
  data.table::dcast(x, sample_id ~ SYMBOL, value.var = "expr")
}

process_dataset <- function(accession, annotation_db) {
  message("Processing ", accession, " ...")

  cel_files <- get_cel_files(file.path(raw_root, accession))
  eset <- affy::rma(affy::ReadAffy(filenames = cel_files))
  expr_sample_probe <- t(Biobase::exprs(eset))
  expr_sample_gene <- collapse_to_gene(expr_sample_probe, annotation_db)

  outfile <- file.path(out_dir, paste0(accession, "_expression.csv"))
  data.table::fwrite(expr_sample_gene, outfile)
  message("Saved: ", outfile)
}

# GSE9103: GPL570, Affymetrix Human Genome U133 Plus 2.0 Array
process_dataset("GSE9103", hgu133plus2.db::hgu133plus2.db)

# GSE15227 and GSE23130: GPL1352, Affymetrix Human X3P Array
process_dataset("GSE15227", u133x3p.db::u133x3p.db)
process_dataset("GSE23130", u133x3p.db::u133x3p.db)

message("Done.")
