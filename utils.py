"""Common utility functions for gene expression analysis."""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List


def load_tcga_data(data_path: str | Path) -> tuple:
    """Load TCGA data and return expression matrix with sample type mask.

    Samples listed in sample.xls whose expression file is not on disk are
    reported and dropped, so an incomplete download still yields the samples
    it does have.
    """
    data_path = Path(data_path)
    sample_path = data_path / 'sample.xls'
    data_path = data_path / 'data'

    sample = pd.read_excel(sample_path)

    found = sample['File Name'].map(lambda name: (data_path / name).is_file())
    if not found.all():
        missing = sample.loc[~found, 'File Name']
        print(f"Skipping {len(missing)} of {len(sample)} samples with no expression file:")
        for name in missing:
            print(f"  - {name}")
        sample = sample[found].reset_index(drop=True)

    if sample.empty:
        raise FileNotFoundError(f"No expression files found in {data_path}")

    normal_mask = sample['Sample Type'] == 'Solid Tissue Normal'

    def get_data(row: pd.Series) -> pd.Series:
        filename = row['File Name']
        file = pd.read_table(data_path / filename, names=['gene_id', 'value'])
        return file['value']

    data = sample.apply(get_data, axis=1)

    filename_0 = sample['File Name'].iloc[0]
    file_0 = pd.read_table(data_path / filename_0, names=['gene_id', 'value'])
    gene_id = file_0.gene_id
    gene_id = np.vectorize(lambda x: x.split('.')[0])(gene_id)
    sample_id = sample['Sample ID']

    data.index = sample_id
    data.columns = gene_id
    normal_mask.index = sample_id

    return data, normal_mask


def load_pathways(pathway_file: str | Path) -> pd.DataFrame:
    """Load pathway data from a parquet file."""
    pathways = pd.read_parquet(pathway_file)
    return pathways


def intersect_data_pathway(bin_data: pd.DataFrame, pathways: pd.DataFrame, genes: np.ndarray) -> tuple:
    """Intersect binary gene data with pathway annotations."""
    pw_genes = pathways.Gene.unique()
    common_genes = np.intersect1d(pw_genes, genes)
    bin_data = bin_data[common_genes]
    pathways = pathways[pathways.Gene.isin(common_genes)]

    return bin_data, pathways, common_genes


def ge2pe_fast(df_expr: pd.DataFrame, pathways: pd.DataFrame, mode: str = "proportion") -> pd.DataFrame:
    """Convert gene expression to pathway expression."""
    pathways_unique = pathways.drop_duplicates(subset=['Gene', 'Pathway'])

    pathway_matrix = pd.crosstab(pathways_unique['Gene'], pathways_unique['Pathway'])

    df_pathway = df_expr.dot(pathway_matrix)

    if mode == "presence":
        df_pathway = (df_pathway > 0).astype(int)
    elif mode == "proportion":
        pw_count = pathway_matrix.sum(axis=0)
        df_pathway = df_pathway.divide(pw_count)

    return df_pathway


def binarize_data(data: pd.DataFrame, genes_above: np.ndarray, genes_bellow: np.ndarray,
                  max_above: pd.Series, min_bellow: pd.Series) -> pd.DataFrame:
    """Binarize gene expression data based on threshold values.

    Assigns 1 if value falls outside the reference range (active), 0 otherwise
    (inactive). Genes classified as "outside" appear in both ``genes_above`` and
    ``genes_bellow``, so the two conditions are combined with OR: overwriting
    would discard the hits found by the first of them.
    """
    bin_above = (data[genes_above] > max_above).astype(int)
    bin_bellow = (data[genes_bellow] < min_bellow).astype(int)

    bin_data = pd.DataFrame(0, columns=data.columns, index=data.index)
    bin_data[bin_above.columns] = bin_above
    bin_data[bin_bellow.columns] |= bin_bellow

    return bin_data


def get_row_genes(row_genes_path: str | Path) -> pd.DataFrame:
    """Load gene name mapping (ensembl to gene symbol)."""
    rg = pd.read_excel(row_genes_path)
    return rg


def get_gene_name_from_ensembl(rg: pd.DataFrame, ensembl_list: List | np.ndarray) -> np.ndarray:
    """Convert ensembl IDs to gene symbols."""
    return rg.set_index('ensembl').loc[ensembl_list].gene_symbol.to_numpy()


def get_ensembl_from_gene_name(rg: pd.DataFrame, gene_list: List | np.ndarray) -> np.ndarray:
    """Convert gene symbols to ensembl IDs."""
    return rg.set_index('gene_symbol').loc[gene_list].ensembl.to_numpy()


def load_ensembl_symbol_map(rg: pd.DataFrame) -> dict:
    """Build an ensembl -> gene symbol lookup.

    Ensembl IDs with no symbol are left out of the mapping, so callers can
    fall back to the ID itself with ``mapping.get(ensembl, ensembl)``.
    """
    valid = rg.dropna(subset=['ensembl', 'gene_symbol'])
    valid = valid[valid.gene_symbol.astype(str).str.strip() != '']

    return dict(zip(valid.ensembl, valid.gene_symbol))
