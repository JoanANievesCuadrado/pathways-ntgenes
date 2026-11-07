"""T-genes analysis pipeline, matplotlib edition.

Mirrors ``../tgenes_analysis.py`` output-for-output, swapping every Plotly
figure for a matplotlib one. All the actual statistics -- T-gene
identification, pathway embedding, MST construction, branch/dendrogram-group
identification, panel-gene loading, CSV aggregation -- are imported directly
from the parent script rather than duplicated; only the plotting functions
(the ones that built a ``go.Figure``) are rewritten here. The dendrogram
itself was already matplotlib-based in the parent script, so it's reused
as-is, legend included.

Writes to ``matplotlib_version/figures and tables/``, never to the parent
project's own ``figures and tables/`` -- this is a parallel, non-destructive
mirror, not a replacement.

Run with the working directory set to the project root (same convention the
parent script uses, since TCGA_FOLDER/EXT_FOLDER are relative paths).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from matplotlib.lines import Line2D

from tgenes_analysis import (
    TCGA_FOLDER,
    BRANCH_COLORS,
    _prepare_tgenes_data,
    _save_gene_frequency,
    _save_least_active,
    _dendrogram,
    _dendrogram_group_labels,
    _save_dendrogram_group_pathways,
    _save_panel_dendrogram_means,
    _load_gene_panel,
    _save_panel_branch_means,
    _build_mst,
    _tree_layout,
    _subtree_sizes,
    _add_virtual_root,
    _branch_identifiers,
    _branch_levels,
    _pathway_commonality_top_down,
    _pathway_commonality_bottom_up,
    _qualifying_branches,
)


OUT_FOLDER = Path(__file__).resolve().parent / 'figures and tables'


# ---------------------------------------------------------------------------
# 1. Pathway distribution
# ---------------------------------------------------------------------------

def _plot_pathway_distribution(pw_tumor, tissue, output_dir):
    """Mean fraction of each pathway's T-genes activated per sample, sorted descending."""
    mean_freq = pw_tumor.mean().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(range(len(mean_freq)), mean_freq.to_numpy(), '-', color='#00A6D6', linewidth=1.5)
    ax.set_title(f'T-Genes Pathway Expression - {tissue}')
    ax.set_xlabel('Pathways (sorted by mean frequency)')
    ax.set_ylabel('Mean fraction of pathway T-genes activated per sample')

    fig.tight_layout()
    fig.savefig(output_dir / 'pathway_distribution.pdf')
    plt.close(fig)
    print("Saved: pathway_distribution.pdf")


# ---------------------------------------------------------------------------
# Activation vs distance scatter
# ---------------------------------------------------------------------------

def _plot_active_pathways_vs_active_tgenes(df, total_tgenes, total_pathways, tissue, output_dir,
                                           cluster_colors, group_labels):
    """Scatter: activated pathways against the number of activated T-genes per sample.

    Points are coloured by ``cluster_colors`` (the dendrogram's leaf colours),
    with a legend naming each group via ``group_labels`` -- the same
    identifier used in the dendrogram-group CSVs.
    """
    counts = df['active_pathways']
    active_tgenes = df['active_tgenes']

    if (counts <= 0).any():
        print(f"  warning: {(counts <= 0).sum()} samples have 0 activated pathways "
              f"and cannot be shown on a log axis")

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.scatter(active_tgenes.to_numpy(), counts.to_numpy(), c=list(cluster_colors),
              s=20, alpha=0.65, edgecolors='none')
    ax.set_yscale('log')
    ax.set_title(f'Activated pathways vs activated T-genes - {tissue}\n{len(df)} tumor samples')
    ax.set_xlabel(f'Activated T-genes (of {total_tgenes})')
    ax.set_ylabel(f'Activated pathways (of {total_pathways})')

    handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=color,
                      markersize=8, label=label) for color, label in group_labels.items()]
    ax.legend(handles=handles, title='Dendrogram group', loc='best')

    fig.tight_layout()
    fig.savefig(output_dir / 'active_pathways_vs_active_tgenes.pdf')
    plt.close(fig)
    print("Saved: active_pathways_vs_active_tgenes.pdf")


# ---------------------------------------------------------------------------
# MST tree and its branch-derived outputs
# ---------------------------------------------------------------------------

def _plot_root_children_sizes(parent, branch_labels, tissue, output_dir):
    """Bar chart of how many samples hang off each direct child of the root."""
    sizes, children, root = _subtree_sizes(parent)
    root_children = sorted(children[root], key=lambda c: -sizes[c])

    x_labels = [branch_labels[c] for c in root_children]
    y_sizes = [int(sizes[c]) for c in root_children]

    print(f"Root's direct children: {len(root_children)}, subtree sizes "
          f"{', '.join(f'{lbl}:{n}' for lbl, n in zip(x_labels, y_sizes))}")

    fig, ax = plt.subplots(figsize=(max(9, 0.6 * len(root_children)), 6))
    bars = ax.bar(x_labels, y_sizes, color='#00A6D6')
    ax.bar_label(bars, padding=3)
    ax.set_title(f'Samples per branch off the root - {tissue}\n'
                f'{len(root_children)} direct children of the root')
    ax.set_xlabel('First son of the root')
    ax.set_ylabel('Samples in that branch (itself + descendants)')
    plt.setp(ax.get_xticklabels(), rotation=90, ha='center')

    fig.tight_layout()
    fig.savefig(output_dir / 'mst_tree_root_children.pdf')
    plt.close(fig)
    print("Saved: mst_tree_root_children.pdf")


def _plot_branch_pathway_commonality(pw_tumor, parent, branch_labels, tissue, output_dir,
                                     branch_samples=20):
    """Common active pathways, walking every large branch off the root from its first node down.

    Alongside the plot, every pathway's mean frequency over the branch's own
    samples, plus its mean frequency over the whole tumor cohort (for
    comparison), is written to its own CSV per branch under a
    ``branch_common_pathways`` subfolder.
    """
    sizes, children, branches = _qualifying_branches(parent, branch_samples)

    if not branches:
        print("  none qualify -- skipping the pathway-commonality plots")
        return

    folder = output_dir / 'branch_common_pathways'
    folder.mkdir(exist_ok=True, parents=True)

    fig, ax = plt.subplots(figsize=(10, 7))

    for branch_i, c in enumerate(branches):
        levels = _branch_levels(children, c)
        results, common = _pathway_commonality_top_down(pw_tumor, levels)
        level_x = [lv for lv, _ in results]
        count_y = [n for _, n in results]

        name = branch_labels[c]
        print(f"  {name} ({sizes[c]} samples): {count_y}")

        color = BRANCH_COLORS[branch_i % len(BRANCH_COLORS)]
        ax.plot(level_x, count_y, '-o', color=color, label=name, markersize=4)

        branch_nodes = [n for level in levels for n in level]
        pathway_names = sorted(pw_tumor.columns)
        branch_freq = pw_tumor.iloc[branch_nodes][pathway_names]
        overall_freq = pw_tumor[pathway_names]

        table = pd.DataFrame({
            'pathway': pathway_names,
            'mean_frequency': branch_freq.mean().to_numpy().round(4),
            'mean_frequency_all_samples': overall_freq.mean().to_numpy().round(4),
        }).sort_values('mean_frequency', ascending=False)

        table.to_csv(folder / f'{name}.csv', index=False)
        print(f"  Saved: branch_common_pathways/{name}.csv ({len(pathway_names)} pathways, "
              f"{len(common)} common to every sample)")

    ax.set_title(f'Common active pathways by branch depth - {tissue}\n'
                f'branches with more than {branch_samples} samples off the root, '
                f'from the first node down')
    ax.set_xlabel("Level (1 = branch's first node)")
    ax.set_ylabel('Active pathways common to every sample up to that level')
    ax.legend(loc='best')

    fig.tight_layout()
    fig.savefig(output_dir / 'mst_branch_pathway_commonality.pdf')
    plt.close(fig)
    print("Saved: mst_branch_pathway_commonality.pdf")


def _plot_branch_pathway_commonality_leaves_up(pw_tumor, parent, branch_labels, tissue,
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

    fig, ax = plt.subplots(figsize=(10, 7))

    for branch_i, c in enumerate(branches):
        levels = _branch_levels(children, c)
        results, _ = _pathway_commonality_bottom_up(pw_tumor, levels)
        step_x = [s for s, _ in results]
        count_y = [n for _, n in results]

        name = branch_labels[c]
        print(f"  {name} ({sizes[c]} samples): {count_y}")

        color = BRANCH_COLORS[branch_i % len(BRANCH_COLORS)]
        ax.plot(step_x, count_y, '-o', color=color, label=name, markersize=4)

    ax.set_title(f'Common active pathways by branch depth - {tissue}\n'
                f'branches with more than {branch_samples} samples off the root, '
                f'from the leaves up')
    ax.set_xlabel("Steps up from the leaves (last step = the branch's first node)")
    ax.set_ylabel('Active pathways common to every sample up to that step')
    ax.legend(loc='best')

    fig.tight_layout()
    fig.savefig(output_dir / 'mst_branch_pathway_commonality_leaves_up.pdf')
    plt.close(fig)
    print("Saved: mst_branch_pathway_commonality_leaves_up.pdf")


def _plot_tree(pw_tumor, tissue, output_dir, branch_samples=20):
    """Draw the MST as a standard top-down tree: x = depth from the root, y from the tree layout.

    Rooted at an artificial sample at the origin of pathway space (see
    ``_add_virtual_root``) instead of the real sample closest to the centre,
    so the root represents a deliberate reference point rather than
    whichever tumor sample happens to be nearest it. Neither axis is pinned
    to a measured quantity: x is just how many edges a node is from the
    root, the usual way a tree gets laid out.
    """
    pw_all, root = _add_virtual_root(pw_tumor)
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
    _plot_branch_pathway_commonality(pw_tumor, parent, branch_labels, tissue, output_dir,
                                     branch_samples)
    _plot_branch_pathway_commonality_leaves_up(pw_tumor, parent, branch_labels, tissue,
                                               output_dir, branch_samples)

    sizes, children, branches = _qualifying_branches(parent, branch_samples)
    branch_color = {branch: BRANCH_COLORS[i % len(BRANCH_COLORS)]
                    for i, branch in enumerate(branches)}

    marker_color = ['#8995A3'] * len(pw_tumor)
    marker_size = [20] * len(pw_tumor)
    for node in range(len(pw_tumor)):
        branch = node
        while parent[branch] >= 0 and parent[branch] != root:
            branch = parent[branch]
        if branch in branch_color:
            marker_color[node] = branch_color[branch]
            marker_size[node] = 36

    fig, ax = plt.subplots(figsize=(14, 8))

    for child in np.flatnonzero(has_parent):
        p = parent[child]
        ax.plot([x[p], x[child]], [y[p], y[child]], color='black', alpha=0.35, linewidth=1,
               zorder=1)

    ax.scatter(x[:len(pw_tumor)], y[:len(pw_tumor)], c=marker_color, s=marker_size, zorder=2)
    ax.scatter([x[root]], [y[root]], marker='D', s=130, color='#d62728',
              edgecolors='black', linewidths=1, zorder=3)

    handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=branch_color[b],
                      markersize=8, label=branch_labels[b]) for b in branches]
    handles.append(Line2D([0], [0], marker='D', color='w', markerfacecolor='#d62728',
                          markeredgecolor='black', markersize=10,
                          label='Root (virtual, zero vector)'))
    handles.append(Line2D([0], [0], marker='o', color='w', markerfacecolor='#8995A3',
                          markersize=8, label='Other samples'))
    ax.legend(handles=handles, title=f'Branch (> {branch_samples} samples)', loc='best')

    ax.set_title(f'Tumor samples - MST tree layout - {tissue}\n'
                f'{len(pw_tumor)} tumor samples, root = {labels[-1]}')
    ax.set_xlabel('Depth from the root')
    ax.set_ylabel('Tree position (not meaningful on its own)')
    ax.set_xticks([])
    ax.set_yticks([])

    fig.tight_layout()
    fig.savefig(output_dir / 'mst_tree.pdf')
    plt.close(fig)
    print("Saved: mst_tree.pdf")

    return parent, branch_labels


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def analyze_tgenes(tissue: str, dendrogram_color_threshold=None, branch_samples=20):
    """Full T-genes pipeline for one tissue, matplotlib output.

    Same signature and same data-preparation pass as
    ``tgenes_analysis.analyze_tgenes`` (imported, not duplicated); only the
    plots differ.
    """
    print(f"Analyzing T-genes for tissue: {tissue}")

    bundle = _prepare_tgenes_data(TCGA_FOLDER / f'TCGA-{tissue}')
    df = bundle['df']
    pw_tumor = bundle['pw_tumor']

    output_dir = OUT_FOLDER / (tissue + '_tgenes')
    output_dir.mkdir(exist_ok=True, parents=True)

    _plot_pathway_distribution(pw_tumor, tissue, output_dir)

    df.to_csv(output_dir / 'activation_vs_distance.csv')
    print(f"Saved: activation_vs_distance.csv ({len(df)} tumor samples)\n")

    _save_gene_frequency(bundle, output_dir, 'tgene_activation_frequency.csv')
    _save_least_active(df, bundle, output_dir)
    print()

    cluster_colors, group_labels = _dendrogram(pw_tumor, tissue, output_dir,
                                               dendrogram_color_threshold)
    _save_dendrogram_group_pathways(pw_tumor, cluster_colors, group_labels, output_dir)
    _plot_active_pathways_vs_active_tgenes(df, bundle['total_tgenes'], bundle['total_pathways'],
                                           tissue, output_dir, cluster_colors, group_labels)

    panel = _load_gene_panel(tissue, 'tgenes.txt', bundle['rg'])
    if panel:
        _save_panel_dendrogram_means(panel, bundle['data'], bundle['normal_mask'], cluster_colors,
                                     group_labels, bundle['gene_labels'], output_dir,
                                     'perfect_panel_tgenes_dendrogram')

    parent, branch_labels = _plot_tree(pw_tumor, tissue, output_dir, branch_samples)

    if panel:
        _save_panel_branch_means(panel, bundle['data'], bundle['normal_mask'], parent,
                                 branch_labels, bundle['gene_labels'], output_dir, branch_samples,
                                 'perfect_panel_tgenes_branch')

    print(f"\nAnalysis complete. Results saved to: {output_dir}")


if __name__ == '__main__':
    analyze_tgenes('PRAD', 55)
    analyze_tgenes('LIHC', 55)
    analyze_tgenes('LUAD', 55)
    analyze_tgenes('GBM', 22)
