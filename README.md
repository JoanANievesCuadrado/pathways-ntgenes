# N- and T-genes framework for pathway deregulation

Code repository for the manuscript:

> **N- and T-genes framework for pathway deregulation: A multi-cancer molecular study of taxonomy, progression, and protection**
> Joan Nieves¹, Dania Vázquez², Augusto González¹
> ¹ Institute of Cybernetics, Mathematics and Physics (ICIMAF), Havana, Cuba
> ² Center for Genetic Engineering and Biotechnology (CIGB), Havana, Cuba

**Status: unpublished, manuscript currently under peer review.** This repository accompanies the submission and is not yet associated with a published article. Please do not redistribute outside the review process.

## Overview

The framework extends the N-gene/T-gene concept (genes with expression intervals exclusive to normal or tumor samples) to pathway-level analysis. For each sample and Reactome pathway, an *effective activation frequency* is computed as the fraction of the pathway's annotated N-genes (in normal samples) or T-genes (in tumor samples) that are active. Three layers of analysis are built on these frequencies:

1. **Cohort averages** — which pathways are important in each tissue.
2. **Hierarchical-clustering (dendrogram) groups**, ordered by number of active genes — sequences of pathway activation/deactivation along progression.
3. **Minimum spanning tree (MST)** over pathway-frequency vectors — a molecular taxonomy independent of gene counts, with branch-level pathway programs.

The framework is applied to four TCGA cohorts — prostate adenocarcinoma (PRAD), lung adenocarcinoma (LUAD), liver hepatocellular carcinoma (LIHC), and glioblastoma (GBM) — analyzing both tumor and normal compartments.

## Repository layout

```
tgenes.py, ngenes.py          Gene-level N-gene / T-gene identification
utils.py                      Shared data loading, binarization, pathway-frequency embedding
tgenes_analysis.py            Full T-genes pipeline for one tissue (Plotly figures)
ngenes_analysis.py            Full N-genes pipeline for one tissue (Plotly figures)
matplotlib_version/           Mirror of the two pipelines above, matplotlib output instead of Plotly
data_external/
  TCGA/TCGA-{TISSUE}/         Per-tissue TCGA expression data + sample sheet (not tracked in git)
  TCGA/rows_genes2.xlsx       Ensembl id <-> gene symbol mapping
  pathways/                   Reactome pathway annotation (All_pathways.parquet, Top_pathways.parquet)
  pathways/perfect_panel/     Curated gene panels per tissue, used for the panel-based figures
figures and tables/           Pipeline outputs (PDF figures + CSV tables), one subfolder per tissue/dataset
```

## Running the pipeline

Scripts assume the working directory is the project root (paths to `data_external/` and `figures and tables/` are relative). From the project root:

```python
from tgenes_analysis import analyze_tgenes
from ngenes_analysis import analyze_ngenes

analyze_tgenes('PRAD', dendrogram_color_threshold=55)
analyze_ngenes('PRAD', dendrogram_color_threshold=30)
```

`tissue` is the TCGA tissue label (e.g. `'PRAD'`); its data is expected at `data_external/TCGA/TCGA-{tissue}/`. `dendrogram_color_threshold` and `branch_samples` are tissue-specific tuning parameters — see the docstrings in `tgenes_analysis.py` / `ngenes_analysis.py` for details, and the bottom of each file for the parameters used for PRAD, LUAD, LIHC, and GBM.

The `matplotlib_version/` subproject exposes the same `analyze_tgenes` / `analyze_ngenes` functions and reuses the data-preparation and statistics code from the root scripts, only re-rendering the figures with matplotlib; its outputs go to `matplotlib_version/figures and tables/` and never touch the root `figures and tables/`.

## Dependencies

Python 3.10+, with `numpy`, `pandas`, `scipy`, `matplotlib`, `plotly`, `openpyxl` (for `rows_genes2.xlsx`), and `pyarrow` (for the pathway `.parquet` files).

## Data availability

Raw TCGA expression data under `data_external/TCGA/` is not tracked in git (see `.gitignore`); it must be obtained from TCGA/GDC separately and placed following the layout above.
