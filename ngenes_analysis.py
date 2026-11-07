"""N-genes analysis pipeline: the complement of ``tgenes_analysis.py``.

Mirrors it role for role -- normal samples instead of tumor samples, T-genes
instead of N-genes as the reference, "distance to the tumor centre" instead
of "distance to the normal centre" -- producing the symmetric set of
outputs: pathway distribution, per-sample activation vs distance to the
tumor centre, and the MST tree structure of normal samples.

Everything comes from one data-preparation pass per tissue
(``_prepare_ngenes_data``). Outputs are PDF/CSV only -- no interactive HTML.

Run different tissues from the same script by calling ``analyze_ngenes``
with different arguments; see the bottom of the file.
"""

import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.colors import qualitative
import scipy.cluster.hierarchy as sch

from matplotlib.colors import to_hex
from matplotlib.lines import Line2D
from pathlib import Path
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import pdist, squareform
from scipy.stats import gmean

from utils import (load_tcga_data, load_pathways, intersect_data_pathway, binarize_data,
                   ge2pe_fast, get_row_genes, load_ensembl_symbol_map)
from ngenes import get_ngenes


BRANCH_COLORS = qualitative.Plotly
from tgenes import get_tgenes


TCGA_FOLDER = Path('data_external/TCGA')
EXT_FOLDER = Path('data_external')
OUT_FOLDER = Path('figures and tables')


# ---------------------------------------------------------------------------
# Shared pathway-space embedding
# ---------------------------------------------------------------------------

def _pathway_positions(sample_data, genes_above, genes_bellow, max_above, min_bellow,
                       genes, pathways_all):
    """Embed samples in pathway-frequency space.

    Returns a samples x pathways frame where entry [s, p] is the fraction of
    the deregulated genes of pathway p that are active in sample s.
    """
    bin_data = binarize_data(sample_data, genes_above, genes_bellow, max_above, min_bellow)
    bin_data, pathways, _ = intersect_data_pathway(bin_data, pathways_all.copy(), genes)

    return ge2pe_fast(bin_data, pathways, mode='proportion')


def _tumor_center(data, normal_mask, pathways_all):
    """Centroid of the tumor samples in T-gene pathway-frequency space.

    The tumor coordinates have to come from the T-genes: those are the genes
    whose tumor expression leaves the normal range, so tumor samples take
    non-zero values along them.
    """
    (tdata, tgenes, tgenes_above, tgenes_bellow,
     tgenes_max_above, tgenes_min_bellow, *_) = get_tgenes(data, normal_mask)

    pw_tumor = _pathway_positions(tdata, tgenes_above, tgenes_bellow,
                                  tgenes_max_above, tgenes_min_bellow,
                                  tgenes, pathways_all)

    print(f"Tumor center: {pw_tumor.shape[0]} samples over {pw_tumor.shape[1]} pathways "
          f"({len(tgenes)} T-genes)")

    return pw_tumor.mean()


def _distance_components(pw_normal, center):
    """Per-pathway distance of every normal sample from the tumor centre.

    Normal positions and the centre are defined over different gene sets, so
    the pathway axes are unioned. A pathway missing on one side has no
    deregulated gene there, which is a coordinate of 0, not a missing value.
    """
    axes = pw_normal.columns.union(center.index)
    shared = len(pw_normal.columns.intersection(center.index))
    print(f"Pathway axes: {len(axes)} total, {shared} shared by T-genes and N-genes")

    positions = pw_normal.reindex(columns=axes).fillna(0.0)
    center = center.reindex(axes).fillna(0.0)

    return (positions - center).abs()


def _distance_to_center(components):
    """Total Euclidean distance of every normal sample to the tumor centre."""
    return np.sqrt((components ** 2).sum(axis=1))


def _tumor_center_genes(data, normal_mask):
    """Centroid of the tumor samples in gene space.

    One coordinate per T-gene: the fraction of tumor samples in which that
    gene is deregulated. This is the gene-space counterpart of the pathway
    centre.
    """
    (tdata, tgenes, tgenes_above, tgenes_bellow,
     tgenes_max_above, tgenes_min_bellow, *_) = get_tgenes(data, normal_mask)

    bin_tdata = binarize_data(tdata, tgenes_above, tgenes_bellow,
                              tgenes_max_above, tgenes_min_bellow)

    print(f"Tumor center in gene space: {bin_tdata.shape[0]} samples "
          f"over {len(tgenes)} T-genes")

    return bin_tdata.mean()


def _gene_distance(bin_ndata, center):
    """Euclidean distance of every normal sample to the tumor centre, in gene space."""
    axes = bin_ndata.columns.union(center.index)
    positions = bin_ndata.reindex(columns=axes).fillna(0.0)
    center = center.reindex(axes).fillna(0.0)

    return np.sqrt(((positions - center) ** 2).sum(axis=1))


def _all_genes_distance(data, normal_mask):
    """Distance of every normal sample to the tumor centre, in all-gene log fold change space.

    The reference is the per-gene geometric mean of the tumor samples, over
    every gene rather than only the N-genes. Because the geometric mean is
    the exponential of the mean log, the tumor centre is the origin of this
    space, so a normal sample's distance to it is the norm of its own
    fold-change vector.
    """
    normal = data[normal_mask] + 0.1
    tumor = data[~normal_mask] + 0.1

    ref = gmean(tumor)
    fc = np.log2(normal / ref)

    print(f"Log fold change over all genes: {fc.shape[0]} normal samples, "
          f"{fc.shape[1]} genes, reference from {tumor.shape[0]} tumors")

    return np.sqrt((fc ** 2).sum(axis=1))


# ---------------------------------------------------------------------------
# Shared data preparation
# ---------------------------------------------------------------------------

def _prepare_ngenes_data(data_path):
    """Load, identify N-genes, and compute everything downstream needs, once.

    Returns a dict bundle: the pathway-distribution plot, the dendrogram, the
    activation-vs-distance outputs, and the MST tree all read from this
    rather than reloading data or recomputing pathway positions.
    """
    data, normal_mask = load_tcga_data(data_path)
    print(f"Data loaded: {data.shape[0]} samples, {data.shape[1]} genes")

    (ndata, ngenes, ngenes_above, ngenes_bellow,
     ngenes_max_above, ngenes_min_bellow,
     ngenes_above_only, ngenes_outside, ngenes_bellow_only) = get_ngenes(data, normal_mask)

    print(f"N-genes identified: {len(ngenes)} total")
    print(f"Tumor samples: {(~normal_mask).sum()}, Normal samples: {normal_mask.sum()}")

    bin_ndata = binarize_data(ndata, ngenes_above, ngenes_bellow,
                              ngenes_max_above, ngenes_min_bellow)

    pathways_all = load_pathways(EXT_FOLDER / 'pathways/All_pathways.parquet')

    # Pathway embedding of the normal samples; feeds the pathway-distribution
    # plot, the dendrogram, the pathway distance, and the MST tree alike.
    pw_normal = _pathway_positions(ndata, ngenes_above, ngenes_bellow,
                                   ngenes_max_above, ngenes_min_bellow,
                                   ngenes, pathways_all)

    # Distance to the tumor centre, in three spaces.
    gene_center = _tumor_center_genes(data, normal_mask)
    gene_distance = _gene_distance(bin_ndata, gene_center)

    all_genes_distance = _all_genes_distance(data, normal_mask)

    center = _tumor_center(data, normal_mask, pathways_all)
    pathway_distance = _distance_to_center(_distance_components(pw_normal, center))

    active_ngenes = bin_ndata.sum(axis=1)
    active_pathways = (pw_normal > 0).sum(axis=1)

    # Which pathway holds which N-gene, for the per-sample exports. The same
    # intersection _pathway_positions performs internally, so the membership
    # matches the frequencies exactly.
    _, pathways_n, _ = intersect_data_pathway(bin_ndata, pathways_all.copy(), ngenes)
    pw_genes_map = (pathways_n.drop_duplicates(subset=['Gene', 'Pathway'])
                              .groupby('Pathway').Gene.apply(list))

    rg = get_row_genes(TCGA_FOLDER / 'rows_genes2.xlsx')
    gene_labels = load_ensembl_symbol_map(rg)

    # Built positionally: TCGA sample ids repeat, and aligning Series on a
    # duplicated index would either raise or silently cross-join the rows.
    # All four come from ndata in the same row order.
    df = pd.DataFrame({
        'gene_distance': gene_distance.to_numpy(),
        'all_genes_distance': all_genes_distance.to_numpy(),
        'pathway_distance': pathway_distance.to_numpy(),
        'active_ngenes': active_ngenes.to_numpy(),
        'active_pathways': active_pathways.to_numpy(),
    }, index=pw_normal.index)

    return {
        'df': df,
        'pw_normal': pw_normal,
        'bin_ndata': bin_ndata,
        'cohort_fraction': bin_ndata.mean(),
        'pw_genes_map': pw_genes_map,
        'pathway_sizes': pathways_all.Pathway.value_counts(),
        'gene_class': {**{g: 'na' for g in ngenes_above_only},
                       **{g: 'no' for g in ngenes_outside},
                       **{g: 'nb' for g in ngenes_bellow_only}},
        'gene_labels': gene_labels,
        'total_ngenes': len(ngenes),
        'total_pathways': pw_normal.shape[1],
        'data': data,
        'normal_mask': normal_mask,
        'rg': rg,
    }


# ---------------------------------------------------------------------------
# 1. Pathway distribution (mean activation frequency per pathway)
# ---------------------------------------------------------------------------

def _plot_pathway_distribution(pw_normal, tissue, output_dir):
    """Mean fraction of each pathway's N-genes activated per sample, sorted descending."""
    mean_freq = pw_normal.mean().sort_values(ascending=False)

    fig = go.Figure(go.Scatter(
        x=list(range(len(mean_freq))),
        y=mean_freq.to_numpy(),
        mode='lines+markers',
        marker=dict(size=6, color='#2CA02C'),
        line=dict(color='#2CA02C', width=2),
    ))

    fig.update_layout(
        title=f'N-Genes Pathway Expression - {tissue}',
        xaxis_title='Pathways (sorted by mean frequency)',
        yaxis_title='Mean fraction of pathway N-genes activated per sample',
        template='plotly_white',
        width=1400,
        height=600,
        showlegend=False,
    )

    fig.write_image(output_dir / 'pathway_distribution.pdf')
    print("Saved: pathway_distribution.pdf")


# ---------------------------------------------------------------------------
# 2-6. Activation vs distance: CSV, least-active samples, dendrogram, N-gene
# frequency, and the pathway-distance scatter plot.
# ---------------------------------------------------------------------------

def _save_gene_frequency(bundle, output_dir, filename):
    """Activation frequency of every N-gene across the normal samples."""
    bin_data = bundle['bin_ndata']
    frequency = bundle['cohort_fraction']
    gene_class = bundle['gene_class']
    gene_labels = bundle['gene_labels']

    genes = list(frequency.index)
    table = pd.DataFrame({
        'gene': [gene_labels.get(g, g) for g in genes],
        'class': [gene_class.get(g, '') for g in genes],
        'frequency': frequency.to_numpy().round(4),
        'n_samples_active': bin_data.sum().to_numpy(),
        'n_samples': bin_data.shape[0],
    }).sort_values('frequency', ascending=False)

    table.to_csv(output_dir / filename, index=False)
    print(f"Saved: {filename} ({len(table)} genes, "
          f"frequency {table.frequency.min():.3f} to {table.frequency.max():.3f})")


def _save_least_active(df, bundle, output_dir, n=10):
    """Write a genes file and a pathways file for each least active sample.

    Takes the ``n`` samples with the fewest active N-genes and the ``n`` with
    the fewest active pathways, then writes their detail into
    ``least_active_samples/`` together with a summary row per sample. The two
    selections usually overlap, so their union is used and each sample
    records the criteria that picked it.

    Everything is addressed positionally: TCGA sample ids repeat, so a label
    lookup would pull several samples at once.
    """
    by_genes = set(np.argsort(df['active_ngenes'].to_numpy(), kind='stable')[:n].tolist())
    by_pathways = set(np.argsort(df['active_pathways'].to_numpy(), kind='stable')[:n].tolist())
    selected = sorted(by_genes | by_pathways)

    folder = output_dir / 'least_active_samples'
    folder.mkdir(exist_ok=True, parents=True)

    print(f"Least active samples: {len(by_genes)} by genes, {len(by_pathways)} by "
          f"pathways, {len(selected)} together")

    bin_data = bundle['bin_ndata']
    pw_data = bundle['pw_normal']
    cohort_fraction = bundle['cohort_fraction']
    gene_class = bundle['gene_class']
    pw_genes_map = bundle['pw_genes_map']
    pathway_sizes = bundle['pathway_sizes']
    gene_labels = bundle['gene_labels']

    repeated = df.index.duplicated(keep=False)
    summary = []

    for i in selected:
        sample = df.index[i]

        # Sample ids repeat in TCGA, so a shared id would have the two samples
        # overwriting each other's files; disambiguate by row position.
        stem = f'{sample}_{i}' if repeated[i] else str(sample)
        stem = re.sub(r'[^A-Za-z0-9._-]', '_', stem)

        if i in by_genes and i in by_pathways:
            reason = 'genes+pathways'
        elif i in by_genes:
            reason = 'genes'
        else:
            reason = 'pathways'

        # --- genes ----------------------------------------------------------
        genes_row = bin_data.iloc[i]
        active_genes = list(genes_row.index[genes_row > 0])

        genes_df = pd.DataFrame({
            'gene': [gene_labels.get(g, g) for g in active_genes],
            'class': [gene_class.get(g, '') for g in active_genes],
            'cohort_fraction': [round(float(cohort_fraction[g]), 4) for g in active_genes],
        }).sort_values('cohort_fraction', ascending=False)
        genes_df.to_csv(folder / f'{stem}_genes.csv', index=False)

        # --- pathways -------------------------------------------------------
        active = set(active_genes)
        pw_row = pw_data.iloc[i]

        records = []
        for pathway in pw_row.index[pw_row > 0]:
            members = pw_genes_map[pathway]
            hits = [g for g in members if g in active]
            records.append({
                'pathway': pathway,
                'frequency': round(float(pw_row[pathway]), 4),
                'active_genes': len(hits),
                'genes_in_pathway': len(members),
                'pathway_size': int(pathway_sizes.get(pathway, 0)),
                'gene_names': ';'.join(gene_labels.get(g, g) for g in hits),
            })

        # Explicit columns: a sample with zero active pathways leaves
        # ``records`` empty, and an empty DataFrame built without them has no
        # 'frequency' column to sort by.
        pw_columns = ['pathway', 'frequency', 'active_genes', 'genes_in_pathway',
                      'pathway_size', 'gene_names']
        pw_df = pd.DataFrame(records, columns=pw_columns).sort_values('frequency', ascending=False)
        pw_df.to_csv(folder / f'{stem}_pathways.csv', index=False)

        summary.append({
            'sample': sample,
            'file_stem': stem,
            'selected_by': reason,
            'n_active_genes': df['active_ngenes'].iloc[i],
            'n_active_pathways': df['active_pathways'].iloc[i],
            'gene_distance': round(float(df['gene_distance'].iloc[i]), 4),
            'all_genes_distance': round(float(df['all_genes_distance'].iloc[i]), 4),
            'pathway_distance': round(float(df['pathway_distance'].iloc[i]), 4),
        })

    pd.DataFrame(summary).to_csv(folder / 'summary.csv', index=False)
    print(f"Saved: least_active_samples/ ({2 * len(selected)} files plus summary.csv)")


def _dendrogram(pw_data, tissue, output_dir, color_threshold=None):
    """Ward dendrogram over the pathway vector of every sample, with a legend
    naming each colour group.

    ``color_threshold`` is the merge-height cut scipy colors the branches
    below differently -- the groups it visually separates are specific to
    each tissue's distance scale, so there's no one default that fits every
    tissue; pass the value tuned for this one.

    Returns the colour scipy gave each sample, ordered by the rows of
    ``pw_data`` (so other plots can colour the same samples the same way),
    together with the ``{tissue}_group_{i}`` label for each colour (see
    ``_dendrogram_group_labels``) -- the same identifier used in the legend
    here and in every other dendrogram-group output.
    """
    linkage_matrix = sch.linkage(pw_data, method='ward')

    fig_height = 4
    fig_width = fig_height * 1.618
    fontsize = 11.5

    plt.figure(figsize=(fig_width, fig_height))
    dendro = sch.dendrogram(linkage_matrix, no_labels=True, color_threshold=color_threshold)
    plt.ylabel('Distances', fontsize=fontsize)
    plt.xlabel('Samples', fontsize=fontsize)
    plt.title(tissue)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)

    # scipy names colours through the matplotlib cycle ("C0", "C1", ...), which
    # plotly cannot read, so resolve them to hex here.
    colors = np.empty(pw_data.shape[0], dtype=object)
    for position, sample in enumerate(dendro['leaves']):
        colors[sample] = to_hex(dendro['leaves_color_list'][position])

    group_labels = _dendrogram_group_labels(colors, tissue)
    legend_handles = [Line2D([0], [0], color=color, lw=2) for color in group_labels]
    plt.legend(legend_handles, list(group_labels.values()), title='Dendrogram group',
              fontsize=fontsize * 0.7, title_fontsize=fontsize * 0.8, loc='best')

    plt.tight_layout()
    plt.savefig(output_dir / 'pathway_dendrogram.pdf')
    plt.close()
    print("Saved: pathway_dendrogram.pdf")

    counts = pd.Series(colors).value_counts()
    print(f"Dendrogram: {len(counts)} leaf colours over {len(colors)} samples "
          f"({', '.join(f'{c}:{n}' for c, n in counts.items())})")

    return colors, group_labels


def _dendrogram_group_labels(cluster_colors, tissue):
    """Map each dendrogram leaf colour to a stable ``{tissue}_group_{i}`` label.

    Groups are numbered by size, largest first, so the label is consistent
    everywhere it's used -- CSVs, filenames, plot legends -- rather than
    every caller deriving its own order (and so its own numbering)
    independently.
    """
    counts = pd.Series(cluster_colors).value_counts()  # largest group first
    return {color: f'{tissue}_group_{i + 1}' for i, color in enumerate(counts.index)}


def _save_dendrogram_group_pathways(pw_normal, cluster_colors, group_labels, output_dir):
    """One CSV per dendrogram group: every pathway's mean frequency in that
    group, alongside its mean frequency over the whole normal cohort.

    Mirrors the branch_common_pathways CSVs (see
    ``_plot_branch_pathway_commonality``), just grouped by dendrogram leaf
    colour instead of MST branch. ``group_labels`` (see
    ``_dendrogram_group_labels``) names each group, so the same identifier
    is used here and in ``_save_panel_dendrogram_means``.
    """
    folder = output_dir / 'dendrogram_group_pathways'
    folder.mkdir(exist_ok=True, parents=True)

    pathway_names = sorted(pw_normal.columns)
    overall_freq = pw_normal[pathway_names]
    overall_mean = overall_freq.mean()

    colors = pd.Series(cluster_colors)
    groups = colors.value_counts()  # largest group first

    print(f"Dendrogram groups: {len(groups)} over {len(colors)} samples")

    for color, size in groups.items():
        group_nodes = colors.index[colors == color].tolist()
        group_freq = pw_normal.iloc[group_nodes][pathway_names]

        table = pd.DataFrame({
            'pathway': pathway_names,
            'mean_frequency': group_freq.mean().to_numpy().round(4),
            'mean_frequency_all_samples': overall_mean.to_numpy().round(4),
        }).sort_values('mean_frequency', ascending=False)

        stem = group_labels[color]
        table.to_csv(folder / f'{stem}.csv', index=False)
        print(f"  Saved: dendrogram_group_pathways/{stem}.csv "
              f"({len(pathway_names)} pathways, {size} samples)")


def _save_panel_dendrogram_means(panel_genes, data, normal_mask, cluster_colors, group_labels,
                                 gene_labels, output_dir, stem):
    """Mean activation frequency and mean log fold change of each panel gene,
    genes x dendrogram groups: one column per group, saved as two separate
    CSVs (``{stem}_activation.csv`` and ``{stem}_log_fc.csv``).

    Mirrors ``_save_panel_branch_means``, just grouped by dendrogram leaf
    colour instead of MST branch -- ``group_labels`` (see
    ``_dendrogram_group_labels``) names each group the same way
    ``_save_dendrogram_group_pathways`` does. Activation uses the same rule
    as N-genes themselves (a normal value outside the tumor range), applied
    to every panel gene regardless of whether it passed the N-gene
    significance threshold. Log fold change is against the same
    tumor-sample geometric-mean reference used elsewhere (see
    ``_all_genes_distance``).
    """
    present = [g for g in panel_genes if g in data.columns]
    missing = [g for g in panel_genes if g not in data.columns]
    if missing:
        print(f"  warning: {len(missing)} panel gene(s) not in the expression data: "
              f"{', '.join(missing)}")
    if not present:
        print(f"  no panel genes present in the data -- skipping {stem}")
        return

    tumor = data[~normal_mask][present]
    normal = data[normal_mask][present]

    active = ((normal > tumor.max()) | (normal < tumor.min())).astype(int)

    ref = gmean(tumor + 0.1)
    fc = np.log2((normal + 0.1) / ref)

    colors = pd.Series(cluster_colors)
    groups = colors.value_counts()  # largest group first

    gene_names = [gene_labels.get(g, g) for g in present]
    activation_table = pd.DataFrame({'gene': gene_names})
    fc_table = pd.DataFrame({'gene': gene_names})
    for color, size in groups.items():
        group_nodes = colors.index[colors == color].tolist()
        label = group_labels[color]
        activation_table[label] = active.iloc[group_nodes].mean().to_numpy().round(4)
        fc_table[label] = fc.iloc[group_nodes].mean().to_numpy().round(4)

    activation_table.to_csv(output_dir / f'{stem}_activation.csv', index=False)
    fc_table.to_csv(output_dir / f'{stem}_log_fc.csv', index=False)
    print(f"Saved: {stem}_activation.csv, {stem}_log_fc.csv "
          f"({len(present)} genes, {len(groups)} dendrogram groups)")


def _plot_active_pathways_vs_active_ngenes(df, total_ngenes, total_pathways, tissue, output_dir,
                                           cluster_colors, group_labels):
    """Scatter: activated pathways against the number of activated N-genes per sample.

    Points are coloured by ``cluster_colors`` (the dendrogram's leaf colours,
    in the same sample order), so the pathway-space clusters the dendrogram
    defines are visible here too, with a legend naming each group via
    ``group_labels`` (see ``_dendrogram_group_labels``) -- the same
    identifier used in the dendrogram-group CSVs. Log y: an activated-pathway
    count spans orders of magnitude across samples, and a log axis simply
    drops non-positive values, so a sample with zero stays silently off the
    plot.
    """
    counts = df['active_pathways']
    active_ngenes = df['active_ngenes']

    if (counts <= 0).any():
        print(f"  warning: {(counts <= 0).sum()} samples have 0 activated pathways "
              f"and cannot be shown on a log axis")

    fig = go.Figure(go.Scattergl(
        x=active_ngenes.to_numpy(),
        y=counts.to_numpy(),
        mode='markers',
        marker=dict(size=6, color=cluster_colors, opacity=0.65),
        showlegend=False,
    ))

    # Legend entries from dummy off-chart points, one per dendrogram group,
    # since the real markers are a single multi-coloured trace.
    for color, label in group_labels.items():
        fig.add_trace(go.Scattergl(
            x=[None], y=[None],
            mode='markers',
            marker=dict(size=8, color=color),
            name=label,
        ))

    fig.update_layout(
        title=f'Activated pathways vs activated N-genes - {tissue}'
              f'<br><sub>{len(df)} normal samples</sub>',
        xaxis_title=f'Activated N-genes (of {total_ngenes})',
        yaxis_title=f'Activated pathways (of {total_pathways})',
        yaxis_type='log',
        template='plotly_white',
        width=1100,
        height=700,
        legend=dict(title='Dendrogram group'),
    )

    fig.write_image(output_dir / 'active_pathways_vs_active_ngenes.pdf')
    print("Saved: active_pathways_vs_active_ngenes.pdf")


# ---------------------------------------------------------------------------
# 7. MST tree of normal samples, x = active N-genes per sample
# ---------------------------------------------------------------------------

def _build_mst(pw_data):
    """Minimum spanning tree over samples, edge weight = pathway-space distance.

    Returns the symmetric adjacency matrix (so either endpoint can be read as
    "the other side of this edge").
    """
    dist = squareform(pdist(pw_data.to_numpy()))
    mst = minimum_spanning_tree(dist).toarray()
    return np.maximum(mst, mst.T)  # csgraph returns a directed/triangular matrix


def _tree_layout(mst, root):
    """y coordinate for every node from an iterative post-order walk of the MST.

    x is supplied by the caller (the real distance to the centre) and never
    touched here. Children of a node are visited in MST order; a leaf gets the
    next free integer slot, and a parent is placed at the mean y of its
    children -- the same rule scipy's own dendrogram uses, just decoupled from
    x. An explicit stack is used instead of recursion so a cohort of several
    hundred normal samples cannot hit Python's recursion limit.
    """
    n = mst.shape[0]
    adjacency = [np.flatnonzero(mst[i]) for i in range(n)]

    y = np.full(n, np.nan)
    parent = np.full(n, -1, dtype=int)
    order = []  # post-order: children appear before their parent

    visited = np.zeros(n, dtype=bool)
    visited[root] = True
    stack = [(root, iter(adjacency[root]))]

    while stack:
        node, children = stack[-1]
        advanced = False
        for child in children:
            if not visited[child]:
                visited[child] = True
                parent[child] = node
                stack.append((child, iter(adjacency[child])))
                advanced = True
                break
        if not advanced:
            order.append(node)
            stack.pop()

    next_leaf_slot = 0
    for node in order:
        children = [c for c in adjacency[node] if parent[c] == node]
        if not children:
            y[node] = next_leaf_slot
            next_leaf_slot += 1
        else:
            y[node] = np.mean([y[c] for c in children])

    return y, parent


def _subtree_sizes(parent):
    """Size of every node's subtree (itself plus every descendant).

    Returns the per-node size array, the children-list-per-node adjacency,
    and the root index (the one node with no parent).
    """
    n = len(parent)
    children = [[] for _ in range(n)]
    root = None
    for node, p in enumerate(parent):
        if p >= 0:
            children[p].append(node)
        else:
            root = node

    order = []
    stack = [(root, iter(children[root]))]
    while stack:
        node, it = stack[-1]
        advanced = False
        for child in it:
            stack.append((child, iter(children[child])))
            advanced = True
            break
        if not advanced:
            order.append(node)
            stack.pop()

    sizes = np.ones(n, dtype=int)
    for node in order:
        for child in children[node]:
            sizes[node] += sizes[child]

    return sizes, children, root


def _add_virtual_root(pw_normal):
    """Append an artificial sample at the origin, to anchor the tree root.

    Every pathway coordinate at 0 -- the origin of the pathway-frequency
    space itself, independent of any group's actual distribution. The tree
    decides on its own which real sample(s) it ends up attached to.

    Returns the augmented pathway frame (the virtual root appended as the
    last row) and its row index.
    """
    root_label = 'Zero vector (virtual root)'
    pw_all = pw_normal.copy()
    pw_all.loc[root_label] = 0.0

    root = len(pw_normal)
    print("Virtual root: zero vector, attaches to its nearest real neighbor(s) in pathway space")

    return pw_all, root


def _branch_identifiers(root_children, tissue):
    """Map each of the root's direct children to a stable ``{tissue}_branch_{i}`` label.

    ``root_children`` must already be ordered largest-first (as
    ``_plot_root_children_sizes`` and ``_qualifying_branches`` both sort
    them), so a branch's number is identical whether it's shown in the
    root-children bar chart or one of the size-filtered per-branch outputs:
    filtering down to a subset never changes the relative rank of whichever
    branches remain.
    """
    return {branch: f'{tissue}_branch_{i + 1}' for i, branch in enumerate(root_children)}


def _plot_root_children_sizes(parent, branch_labels, tissue, output_dir):
    """Bar chart of how many samples hang off each direct child of the root."""
    sizes, children, root = _subtree_sizes(parent)
    root_children = sorted(children[root], key=lambda c: -sizes[c])

    x_labels = [branch_labels[c] for c in root_children]
    y_sizes = [int(sizes[c]) for c in root_children]

    print(f"Root's direct children: {len(root_children)}, subtree sizes "
          f"{', '.join(f'{lbl}:{n}' for lbl, n in zip(x_labels, y_sizes))}")

    fig = go.Figure(go.Bar(
        x=x_labels,
        y=y_sizes,
        marker=dict(color='#2CA02C'),
        text=y_sizes,
        textposition='outside',
    ))

    fig.update_layout(
        title=f'Samples per branch off the root - {tissue}'
              f'<br><sub>{len(root_children)} direct children of the root</sub>',
        xaxis_title='First son of the root',
        yaxis_title='Samples in that branch (itself + descendants)',
        template='plotly_white',
        width=max(900, 60 * len(root_children)),
        height=600,
        showlegend=False,
    )

    fig.write_image(output_dir / 'mst_tree_root_children.pdf')
    print("Saved: mst_tree_root_children.pdf")


def _active_pathways(pw_normal, node):
    """Set of pathways active in one real sample (frequency > 0), by position."""
    row = pw_normal.iloc[node]
    return set(row.index[row > 0])


def _branch_levels(children, first_node):
    """Group a branch's nodes into BFS depth layers from its first node.

    Level 1 is always just ``[first_node]``; level k+1 is its tree
    descendants k edges below it. Returns a list of per-level node-index
    lists, ending at the branch's deepest layer.
    """
    levels = [[first_node]]
    while True:
        next_nodes = [c for node in levels[-1] for c in children[node]]
        if not next_nodes:
            break
        levels.append(next_nodes)

    return levels


def _pathway_commonality_top_down(pw_normal, levels):
    """Running intersection of active pathways, walking a branch from its first node down.

    Returns a list of (level, count) pairs together with the final common
    set: the pathways active in every single sample of the branch.
    """
    common = _active_pathways(pw_normal, levels[0][0])
    results = [(1, len(common))]

    for level_num, nodes in enumerate(levels[1:], start=2):
        level_common = set.intersection(*(_active_pathways(pw_normal, n) for n in nodes))
        common &= level_common
        results.append((level_num, len(common)))

    return results, common


def _pathway_commonality_bottom_up(pw_normal, levels):
    """Running intersection of active pathways, walking a branch from its leaves up.

    The mirror of ``_pathway_commonality_top_down``: the same BFS layers,
    folded in from the branch's deepest layer toward its first node.
    """
    reversed_levels = levels[::-1]

    common = set.intersection(*(_active_pathways(pw_normal, n) for n in reversed_levels[0]))
    results = [(1, len(common))]

    for step, nodes in enumerate(reversed_levels[1:], start=2):
        level_common = set.intersection(*(_active_pathways(pw_normal, n) for n in nodes))
        common &= level_common
        results.append((step, len(common)))

    return results, common


def _qualifying_branches(parent, branch_samples):
    """Direct children of the root with more than ``branch_samples`` total samples."""
    sizes, children, root = _subtree_sizes(parent)
    branches = sorted((c for c in children[root] if sizes[c] > branch_samples),
                      key=lambda c: -sizes[c])

    print(f"Branches with more than {branch_samples} samples: {len(branches)} of "
          f"{len(children[root])} total")

    return sizes, children, branches


def _plot_branch_pathway_commonality(pw_normal, parent, branch_labels, tissue, output_dir,
                                     branch_samples=20):
    """Common active pathways, walking every large branch off the root from its first node down.

    Alongside the plot, every pathway's mean and std frequency over the
    branch's own samples, plus its mean frequency over the whole normal
    cohort (for comparison), is written to its own CSV per branch under a
    ``branch_common_pathways`` subfolder. ``branch_labels`` (see
    ``_branch_identifiers``) names each branch, so the same identifier is
    used here and in ``_save_panel_branch_means``.
    """
    sizes, children, branches = _qualifying_branches(parent, branch_samples)

    if not branches:
        print("  none qualify -- skipping the pathway-commonality plots")
        return

    folder = output_dir / 'branch_common_pathways'
    folder.mkdir(exist_ok=True, parents=True)

    fig = go.Figure()

    for branch_i, c in enumerate(branches):
        levels = _branch_levels(children, c)
        results, common = _pathway_commonality_top_down(pw_normal, levels)
        level_x = [lv for lv, _ in results]
        count_y = [n for _, n in results]

        name = branch_labels[c]
        print(f"  {name} ({sizes[c]} samples): {count_y}")

        fig.add_trace(go.Scatter(
            x=level_x, y=count_y,
            mode='lines+markers',
            name=name,
            line=dict(color=BRANCH_COLORS[branch_i % len(BRANCH_COLORS)]),
            marker=dict(color=BRANCH_COLORS[branch_i % len(BRANCH_COLORS)]),
        ))

        branch_nodes = [n for level in levels for n in level]
        pathway_names = sorted(pw_normal.columns)
        branch_freq = pw_normal.iloc[branch_nodes][pathway_names]
        overall_freq = pw_normal[pathway_names]

        table = pd.DataFrame({
            'pathway': pathway_names,
            'mean_frequency': branch_freq.mean().to_numpy().round(4),
            'mean_frequency_all_samples': overall_freq.mean().to_numpy().round(4),
        }).sort_values('mean_frequency', ascending=False)

        stem = name
        table.to_csv(folder / f'{stem}.csv', index=False)
        print(f"  Saved: branch_common_pathways/{stem}.csv ({len(pathway_names)} pathways, "
              f"{len(common)} common to every sample)")

    fig.update_layout(
        title=f'Common active pathways by branch depth - {tissue}'
              f'<br><sub>branches with more than {branch_samples} samples off the root, '
              f'from the first node down</sub>',
        xaxis_title="Level (1 = branch's first node)",
        yaxis_title='Active pathways common to every sample up to that level',
        template='plotly_white',
        width=1000,
        height=700,
    )

    fig.write_image(output_dir / 'mst_branch_pathway_commonality.pdf')
    print("Saved: mst_branch_pathway_commonality.pdf")


def _plot_branch_pathway_commonality_leaves_up(pw_normal, parent, branch_labels, tissue,
                                               output_dir, branch_samples=20):
    """Mirror of ``_plot_branch_pathway_commonality``, walking each branch from its leaves up.

    Every curve ends at the same full-branch common count the top-down plot
    does, just reaches it along a different trajectory. The common-pathway
    CSVs are not rewritten here: the final intersection does not depend on
    the order it was taken in.
    """
    sizes, children, branches = _qualifying_branches(parent, branch_samples)

    if not branches:
        print("  none qualify -- skipping the leaves-up pathway-commonality plot")
        return

    fig = go.Figure()

    for branch_i, c in enumerate(branches):
        levels = _branch_levels(children, c)
        results, _ = _pathway_commonality_bottom_up(pw_normal, levels)
        step_x = [s for s, _ in results]
        count_y = [n for _, n in results]

        name = branch_labels[c]
        print(f"  {name} ({sizes[c]} samples): {count_y}")

        fig.add_trace(go.Scatter(
            x=step_x, y=count_y,
            mode='lines+markers',
            name=name,
            line=dict(color=BRANCH_COLORS[branch_i % len(BRANCH_COLORS)]),
            marker=dict(color=BRANCH_COLORS[branch_i % len(BRANCH_COLORS)]),
        ))

    fig.update_layout(
        title=f'Common active pathways by branch depth - {tissue}'
              f'<br><sub>branches with more than {branch_samples} samples off the root, '
              f'from the leaves up</sub>',
        xaxis_title="Steps up from the leaves (last step = the branch's first node)",
        yaxis_title='Active pathways common to every sample up to that step',
        template='plotly_white',
        width=1000,
        height=700,
    )

    fig.write_image(output_dir / 'mst_branch_pathway_commonality_leaves_up.pdf')
    print("Saved: mst_branch_pathway_commonality_leaves_up.pdf")


def _load_gene_panel(tissue, filename, rg):
    """Gene panel for one tissue, resolved from symbols/Ensembl IDs to Ensembl IDs.

    The panel is opt-in, tissue by tissue: returns None if no panel file
    exists yet for this tissue. Entries already shaped like an Ensembl ID
    (``ENSG...``) are kept as-is; everything else is looked up by gene
    symbol, and unmatched symbols are reported and dropped.
    """
    path = EXT_FOLDER / 'pathways' / 'perfect_panel' / tissue / filename
    if not path.exists():
        return None

    names = [line.strip() for line in path.read_text().splitlines() if line.strip()]

    symbol_map = rg.set_index('gene_symbol')['ensembl']
    resolved, missing = [], []
    for name in names:
        if name.startswith('ENSG'):
            resolved.append(name)
        elif name in symbol_map.index:
            value = symbol_map.loc[name]
            value = value.iloc[0] if hasattr(value, 'iloc') else value
            if pd.isna(value):
                missing.append(name)
            else:
                resolved.append(value)
        else:
            missing.append(name)

    if missing:
        print(f"  warning: {len(missing)} panel gene(s) not found by symbol: {', '.join(missing)}")

    print(f"Perfect panel ({filename}): {len(resolved)} of {len(names)} gene(s) resolved")
    return resolved


def _save_panel_branch_means(panel_genes, data, normal_mask, parent, branch_labels, gene_labels,
                             output_dir, branch_samples, stem):
    """Mean activation frequency and mean log fold change of each panel gene,
    genes x branches: one column per MST branch, saved as two separate CSVs
    (``{stem}_activation.csv`` and ``{stem}_log_fc.csv``).

    Activation uses the same rule as N-genes themselves (a normal value
    outside the tumor range), just applied to every panel gene regardless of
    whether it passed the N-gene significance threshold. Log fold change is
    against the same tumor-sample geometric-mean reference used elsewhere
    (see ``_all_genes_distance``).
    """
    _, children, branches = _qualifying_branches(parent, branch_samples)
    if not branches:
        print(f"  none qualify -- skipping {stem}")
        return

    present = [g for g in panel_genes if g in data.columns]
    missing = [g for g in panel_genes if g not in data.columns]
    if missing:
        print(f"  warning: {len(missing)} panel gene(s) not in the expression data: "
              f"{', '.join(missing)}")
    if not present:
        print(f"  no panel genes present in the data -- skipping {stem}")
        return

    tumor = data[~normal_mask][present]
    normal = data[normal_mask][present]

    active = ((normal > tumor.max()) | (normal < tumor.min())).astype(int)

    ref = gmean(tumor + 0.1)
    fc = np.log2((normal + 0.1) / ref)

    gene_names = [gene_labels.get(g, g) for g in present]
    activation_table = pd.DataFrame({'gene': gene_names})
    fc_table = pd.DataFrame({'gene': gene_names})
    for c in branches:
        branch_nodes = [n for level in _branch_levels(children, c) for n in level]
        label = branch_labels[c]
        activation_table[label] = active.iloc[branch_nodes].mean().to_numpy().round(4)
        fc_table[label] = fc.iloc[branch_nodes].mean().to_numpy().round(4)

    activation_table.to_csv(output_dir / f'{stem}_activation.csv', index=False)
    fc_table.to_csv(output_dir / f'{stem}_log_fc.csv', index=False)
    print(f"Saved: {stem}_activation.csv, {stem}_log_fc.csv "
          f"({len(present)} genes, {len(branches)} branches)")


def _plot_tree(pw_normal, tissue, output_dir, branch_samples=20):
    """Draw the MST as a standard top-down tree: x = depth from the root, y from the tree layout.

    Rooted at an artificial sample at the origin of pathway space (see
    ``_add_virtual_root``) instead of the real sample closest to the centre,
    so the root represents a deliberate reference point rather than
    whichever normal sample happens to be nearest it. Neither axis is
    pinned to a measured quantity: x is just how many edges a node is from
    the root, the usual way a tree gets laid out.
    """
    pw_all, root = _add_virtual_root(pw_normal)
    mst = _build_mst(pw_all)

    y, parent = _tree_layout(mst, root)
    labels = pw_all.index

    has_parent = parent >= 0
    print(f"Edges: {has_parent.sum()} total")

    # Depth (edges from the root), via a top-down walk from the same
    # children adjacency _subtree_sizes already builds.
    sizes_all, children_of, _ = _subtree_sizes(parent)
    x = np.zeros(len(parent))
    stack = [root]
    while stack:
        node = stack.pop()
        for child in children_of[node]:
            x[child] = x[node] + 1
            stack.append(child)

    # One stable {tissue}_branch_{i} identifier per direct child of the
    # root, numbered largest-first -- used everywhere a branch is named.
    root_children = sorted(children_of[root], key=lambda c: -sizes_all[c])
    branch_labels = _branch_identifiers(root_children, tissue)

    _plot_root_children_sizes(parent, branch_labels, tissue, output_dir)
    _plot_branch_pathway_commonality(pw_normal, parent, branch_labels, tissue, output_dir,
                                     branch_samples)
    _plot_branch_pathway_commonality_leaves_up(pw_normal, parent, branch_labels, tissue,
                                               output_dir, branch_samples)

    sizes, children, branches = _qualifying_branches(parent, branch_samples)
    branch_color = {branch: BRANCH_COLORS[i % len(BRANCH_COLORS)]
                    for i, branch in enumerate(branches)}
    edge_x, edge_y = [], []
    for child in np.flatnonzero(has_parent):
        p = parent[child]
        edge_x += [x[p], x[child], None]
        edge_y += [y[p], y[child], None]

    fig = go.Figure()

    fig.add_trace(go.Scattergl(
        x=edge_x, y=edge_y,
        mode='lines',
        line=dict(color='rgba(0,0,0,0.35)', width=1),
        showlegend=False,
    ))

    marker_color = ['#8995A3'] * len(pw_normal)
    marker_size = [6] * len(pw_normal)
    for node in range(len(pw_normal)):
        branch = node
        while parent[branch] >= 0 and parent[branch] != root:
            branch = parent[branch]
        if branch in branch_color:
            marker_color[node] = branch_color[branch]
            marker_size[node] = 8

    fig.add_trace(go.Scattergl(
        x=x[:len(pw_normal)], y=y[:len(pw_normal)],
        mode='markers',
        marker=dict(size=marker_size, color=marker_color),
        showlegend=False,
    ))

    fig.add_trace(go.Scattergl(
        x=[x[root]], y=[y[root]], mode='markers',
        marker=dict(size=13, color='#d62728', symbol='diamond',
                    line=dict(color='black', width=1)),
        showlegend=False,
    ))

    # Legend entries are built from dummy off-chart points, one per branch
    # colour, since the real markers are drawn in a single multi-coloured
    # trace (showlegend=False above) rather than one trace per branch.
    for branch in branches:
        fig.add_trace(go.Scattergl(
            x=[None], y=[None],
            mode='markers',
            marker=dict(size=8, color=branch_color[branch]),
            name=branch_labels[branch],
        ))

    fig.add_trace(go.Scattergl(
        x=[None], y=[None],
        mode='markers',
        marker=dict(size=13, color='#d62728', symbol='diamond',
                    line=dict(color='black', width=1)),
        name='Root (virtual, zero vector)',
    ))

    fig.add_trace(go.Scattergl(
        x=[None], y=[None],
        mode='markers',
        marker=dict(size=6, color='#8995A3'),
        name='Other samples',
    ))

    fig.update_layout(
        title=f'Normal samples - MST tree layout - {tissue}'
              f'<br><sub>{len(pw_normal)} normal samples, root = {labels[-1]}</sub>',
        xaxis_title='Depth from the root',
        yaxis_title='Tree position (not meaningful on its own)',
        xaxis=dict(showticklabels=False),
        yaxis=dict(showticklabels=False),
        template='plotly_white',
        width=1400,
        height=800,
        legend=dict(title='Branch (> {} samples)'.format(branch_samples)),
    )

    fig.write_image(output_dir / 'mst_tree.pdf')
    # fig.write_html(output_dir / 'mst_tree.html')
    print("Saved: mst_tree.pdf")

    return parent, branch_labels


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def analyze_ngenes(tissue: str, dendrogram_color_threshold=None, branch_samples=20):
    """Full N-genes pipeline for one tissue: pathway distribution, activation
    vs distance, and the MST tree, from one shared data-preparation pass.

    ``tissue`` is the TCGA tissue label (e.g. ``'PRAD'``); its data folder is
    ``TCGA_FOLDER / f'TCGA-{tissue}'``.

    ``dendrogram_color_threshold`` (the merge-height cut that splits the
    dendrogram into colored groups) and ``branch_samples`` (the minimum size
    for an MST branch to get its own pathway-commonality curve) are both
    tissue-specific -- the defaults are starting points, tune them per
    tissue once you've seen the dendrogram/MST for it.
    """
    print(f"Analyzing N-genes for tissue: {tissue}")

    bundle = _prepare_ngenes_data(TCGA_FOLDER / f'TCGA-{tissue}')
    df = bundle['df']
    pw_normal = bundle['pw_normal']

    output_dir = OUT_FOLDER / (tissue + '_ngenes')
    output_dir.mkdir(exist_ok=True, parents=True)

    _plot_pathway_distribution(pw_normal, tissue, output_dir)

    df.to_csv(output_dir / 'activation_vs_distance.csv')
    print(f"Saved: activation_vs_distance.csv ({len(df)} normal samples)\n")

    _save_gene_frequency(bundle, output_dir, 'ngene_activation_frequency.csv')
    _save_least_active(df, bundle, output_dir)
    print()

    cluster_colors, group_labels = _dendrogram(pw_normal, tissue, output_dir,
                                               dendrogram_color_threshold)
    _save_dendrogram_group_pathways(pw_normal, cluster_colors, group_labels, output_dir)
    _plot_active_pathways_vs_active_ngenes(df, bundle['total_ngenes'], bundle['total_pathways'],
                                           tissue, output_dir, cluster_colors, group_labels)

    panel = _load_gene_panel(tissue, 'ngenes.txt', bundle['rg'])
    if panel:
        _save_panel_dendrogram_means(panel, bundle['data'], bundle['normal_mask'], cluster_colors,
                                     group_labels, bundle['gene_labels'], output_dir,
                                     'perfect_panel_ngenes_dendrogram')

    parent, branch_labels = _plot_tree(pw_normal, tissue, output_dir, branch_samples)

    if panel:
        _save_panel_branch_means(panel, bundle['data'], bundle['normal_mask'], parent,
                                 branch_labels, bundle['gene_labels'], output_dir, branch_samples,
                                 'perfect_panel_ngenes_branch')

    print(f"\nAnalysis complete. Results saved to: {output_dir}")


if __name__ == '__main__':
    analyze_ngenes('PRAD', 30)
    analyze_ngenes('LUAD', branch_samples=10)
    analyze_ngenes('LIHC', branch_samples=10)
    analyze_ngenes('GBM', 25)
