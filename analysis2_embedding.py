"""
BST 281 Final Project — Module 1, Analysis 2
Normalization, embedding, clustering, and cell-type label validation

Author  : Jenny Yang
Input   : adata_filtered.h5ad produced by analysis1_qc.py

Usage:
    python analysis2_embedding.py \
        --adata   results/analysis1/adata_filtered.h5ad \
        --outdir  results/analysis2

Outputs (all saved to --outdir):
    fig_umap_celltype.png         - UMAP coloured by provided cell-type label
    fig_umap_donor.png            - UMAP coloured by donor
    fig_umap_disease.png          - UMAP coloured by disease group
    fig_umap_cluster.png          - UMAP coloured by Leiden cluster
    fig_markers_dotplot.png       - Dot plot: canonical marker genes × cell types
    fig_markers_violin.png        - Violin plots of key markers per cell type
    fig_marker_heatmap.png        - Heatmap: mean marker expression per cluster
    fig_donor_mixing.png          - Silhouette / donor-mixing bar chart per cell type
    table_nuclei_per_donor_celltype.csv  - Nuclei counts donor × cell type
    table_cluster_label_concordance.csv - Fraction of each cluster per cell-type label
    adata_embedded.h5ad           - Final annotated AnnData with UMAP + clusters
"""

import argparse
import os
import random
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import anndata as ad
import scanpy as sc
from sklearn.metrics import silhouette_samples

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
sc.settings.seed = SEED

# ── Pipeline parameters ───────────────────────────────────────────────────────
N_HVG          = 2_000
N_PCA          = 50
N_NEIGHBORS    = 20
LEIDEN_RES     = 0.5    # can be adjusted
FIG_DPI        = 150

# ── Canonical AD-relevant marker genes per cell type ─────────────────────────
MARKER_GENES = {
    "Neuron"         : ["SYT1", "RBFOX3", "SNAP25", "GAD1", "GAD2", "NRGN"],
    "Astrocyte"      : ["GFAP", "AQP4", "ALDH1L1", "SLC1A2", "GJA1"],
    "Microglia"      : ["P2RY12", "CX3CR1", "TMEM119", "IBA1", "CSF1R"],
    "Oligodendrocyte": ["MBP", "MOG", "OLIG2", "PLP1", "MAG"],
    "OPC"            : ["PDGFRA", "CSPG4", "SOX10"],
    "Endothelial"    : ["CLDN5", "FLT1", "ERG"],
}
ALL_MARKERS = [g for gs in MARKER_GENES.values() for g in gs]

PALETTE_CELL  = "tab20"
PALETTE_DONOR = "Set1"
PALETTE_DIS   = {"AD": "#E07B54", "Control": "#5B8DB8",
                 "AD ": "#E07B54", "Control ": "#5B8DB8"}   # handle trailing spaces


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def harmonise_obs(adata):
    """Ensure 'celltype', 'donor', 'disease' columns exist in adata.obs."""
    rename = {}
    col_lower = {c.lower(): c for c in adata.obs.columns}
    candidates = {
        "celltype": ["celltype", "cell_type", "cell.type",
                     "annotation", "cluster_label", "seurat_clusters",
                     "leiden", "louvain", "class"],
        "donor"   : ["donor", "sample", "donor_id", "sampleid", "patient"],
        "disease" : ["disease", "diagnosis", "group", "condition",
                     "disease_group", "ad_ctrl"],
    }
    for target, opts in candidates.items():
        for opt in opts:
            if opt in col_lower and target not in adata.obs.columns:
                rename[col_lower[opt]] = target
    adata.obs = adata.obs.rename(columns=rename)

    # Fallback: if celltype still missing, use "Unknown"
    if "celltype" not in adata.obs.columns:
        print("  Warning: no cell-type column found; "
              "setting celltype='Unknown' for all cells.")
        adata.obs["celltype"] = "Unknown"
    return adata


def filter_markers_present(adata, marker_dict):
    """Return a filtered marker dict containing only genes present in adata."""
    present = set(adata.var_names)
    filtered = {}
    for ct, genes in marker_dict.items():
        kept = [g for g in genes if g in present]
        if kept:
            filtered[ct] = kept
    missing = [g for gs in marker_dict.values() for g in gs if g not in present]
    if missing:
        print(f"  Markers not found in dataset (skipped): {missing}")
    return filtered


def safe_colour_map(categories, palette):
    cats = sorted(set(categories))
    if isinstance(palette, str):
        cmap = plt.get_cmap(palette)
        colors = [mcolors.to_hex(cmap(i / max(len(cats) - 1, 1)))
                  for i in range(len(cats))]
    else:
        colors = list(palette)
    return {c: colors[i % len(colors)] for i, c in enumerate(cats)}


# ─────────────────────────────────────────────────────────────────────────────
# Normalisation & dimensionality reduction
# ─────────────────────────────────────────────────────────────────────────────

def run_preprocessing(adata):
    """
    Primary: log-normalisation (NormalizeData × 10,000 + log1p)
    Secondary: SCTransform-equivalent via scanpy's pp.normalize_total +
               pp.log1p, then highly variable genes.
    Stores raw counts in adata.layers['counts'] before normalising.
    """
    # Preserve raw counts
    adata.layers["counts"] = adata.X.copy()

    print("  Normalising (log-normalisation, scale_factor=10,000) ...")
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata.layers["lognorm"] = adata.X.copy()

    print(f"  Selecting top {N_HVG} highly variable genes ...")
    sc.pp.highly_variable_genes(adata, n_top_genes=N_HVG, flavor="seurat",
                                 batch_key="donor" if "donor" in adata.obs.columns else None)
    print(f"    HVGs found: {adata.var['highly_variable'].sum():,}")

    # Scale only HVGs (clip to max 10)
    adata_hvg = adata[:, adata.var["highly_variable"]].copy()
    sc.pp.scale(adata_hvg, max_value=10)

    print(f"  PCA ({N_PCA} components) ...")
    sc.tl.pca(adata_hvg, n_comps=N_PCA, random_state=SEED)
    # Copy PCA embedding back
    adata.obsm["X_pca"] = adata_hvg.obsm["X_pca"]
    adata.uns["pca"]    = adata_hvg.uns.get("pca", {})

    print(f"  k-NN graph (k={N_NEIGHBORS}) ...")
    sc.pp.neighbors(adata, n_neighbors=N_NEIGHBORS, n_pcs=N_PCA,
                    random_state=SEED, use_rep="X_pca")

    print("  UMAP ...")
    sc.tl.umap(adata, random_state=SEED)

    print(f"  Leiden clustering (resolution={LEIDEN_RES}) ...")
    sc.tl.leiden(adata, resolution=LEIDEN_RES, random_state=SEED,
                 key_added="leiden")

    return adata


# ─────────────────────────────────────────────────────────────────────────────
# UMAP plots
# ─────────────────────────────────────────────────────────────────────────────

def _umap_scatter(adata, colour_by, title, palette, outpath, legend_loc="right margin"):
    fig, ax = plt.subplots(figsize=(7, 6))
    sc.pl.umap(adata, color=colour_by, title=title,
               palette=palette, legend_loc=legend_loc,
               frameon=False, show=False, ax=ax)
    plt.tight_layout()
    fig.savefig(outpath, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {outpath}")


def plot_umaps(adata, outdir):
    _umap_scatter(adata, "celltype",  "Cell-type annotation",
                  PALETTE_CELL,  os.path.join(outdir, "fig_umap_celltype.png"))
    _umap_scatter(adata, "donor",     "Donor",
                  PALETTE_DONOR, os.path.join(outdir, "fig_umap_donor.png"))
    _umap_scatter(adata, "disease",   "Disease group",
                  list(PALETTE_DIS.values())[:adata.obs["disease"].nunique()],
                  os.path.join(outdir, "fig_umap_disease.png"))
    _umap_scatter(adata, "leiden",    "Leiden clusters",
                  PALETTE_CELL,  os.path.join(outdir, "fig_umap_cluster.png"))


# ─────────────────────────────────────────────────────────────────────────────
# Marker gene plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_marker_dotplot(adata, marker_dict, outpath):
    """Dot plot: rows = cell types, cols = marker genes."""
    all_m = [g for gs in marker_dict.values() for g in gs]
    # Order cells by celltype
    adata_ord = adata.copy()
    adata_ord.obs["celltype"] = pd.Categorical(
        adata_ord.obs["celltype"],
        categories=sorted(adata_ord.obs["celltype"].unique())
    )
    fig, ax = plt.subplots(figsize=(max(10, len(all_m) * 0.55), 6))
    sc.pl.dotplot(adata_ord, var_names=marker_dict, groupby="celltype",
                  standard_scale="var", colorbar_title="Mean expr\n(scaled)",
                  show=False, ax=ax)
    plt.suptitle("Canonical marker gene expression per annotated cell type", y=1.02)
    plt.tight_layout()
    fig.savefig(outpath, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {outpath}")


def plot_marker_violin(adata, marker_dict, outpath, max_genes=12):
    """Violin plots for key marker genes split by cell type."""
    all_m = [g for gs in marker_dict.values() for g in gs][:max_genes]
    n = len(all_m)
    ncols = 4
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3))
    axes = axes.flatten()

    celltypes = sorted(adata.obs["celltype"].unique())
    color_map  = safe_colour_map(celltypes, PALETTE_CELL)

    for i, gene in enumerate(all_m):
        ax = axes[i]
        data = [adata[adata.obs["celltype"] == ct, gene].X.toarray().flatten()
                if hasattr(adata[:, gene].X, "toarray")
                else adata[adata.obs["celltype"] == ct, gene].X.flatten()
                for ct in celltypes]
        parts = ax.violinplot(data, positions=range(len(celltypes)),
                              showmedians=True, showextrema=False)
        for j, pc in enumerate(parts["bodies"]):
            pc.set_facecolor(color_map[celltypes[j]])
            pc.set_alpha(0.75)
        ax.set_xticks(range(len(celltypes)))
        ax.set_xticklabels(celltypes, rotation=50, ha="right", fontsize=7)
        ax.set_title(gene, fontsize=9, fontstyle="italic")
        ax.set_ylabel("log-norm expr", fontsize=7)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Marker gene expression per cell type", fontsize=12, y=1.01)
    plt.tight_layout()
    fig.savefig(outpath, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {outpath}")


def plot_marker_heatmap(adata, marker_dict, outpath):
    """Heatmap: mean scaled expression of marker genes × Leiden cluster."""
    all_m = [g for gs in marker_dict.values() for g in gs]
    # Compute mean expression per Leiden cluster
    clusters = sorted(adata.obs["leiden"].unique(), key=lambda x: int(x))
    rows = {}
    for cl in clusters:
        sub = adata[adata.obs["leiden"] == cl, all_m]
        vals = np.array(sub.X.mean(axis=0)).flatten()
        rows[cl] = vals
    # genes as rows, clusters as columns
    mean_expr = pd.DataFrame(rows, index=all_m)   # shape: (n_genes, n_clusters)

    # z-score each gene (row) across clusters — returns same shape DataFrame
    from scipy.stats import zscore as _zscore
    mean_z = mean_expr.apply(
        lambda row: pd.Series(_zscore(row.values, ddof=1), index=row.index)
                   if row.std() > 0
                   else row,
        axis=1
    )

    fig, ax = plt.subplots(figsize=(max(8, len(clusters) * 0.7),
                                    max(6, len(all_m) * 0.35)))
    sns.heatmap(mean_z, cmap="RdBu_r", center=0, linewidths=0.3,
                ax=ax, yticklabels=True,
                cbar_kws={"label": "z-scored mean expr"})
    ax.set_xlabel("Leiden cluster", fontsize=10)
    ax.set_title("Marker gene mean expression per Leiden cluster\n"
                 "(z-scored across clusters per gene)", fontsize=11)
    plt.tight_layout()
    fig.savefig(outpath, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {outpath}")


# ─────────────────────────────────────────────────────────────────────────────
# Donor-mixing / silhouette score
# ─────────────────────────────────────────────────────────────────────────────

def compute_donor_mixing(adata, outpath):
    """
    Silhouette score per cell type: how well-separated are donors within
    each cell type in PCA space?  A high score = strong donor-specific
    structure (may need batch correction).
    """
    results = []
    pca_coords = adata.obsm["X_pca"]
    celltypes  = adata.obs["celltype"].unique()

    for ct in celltypes:
        mask = adata.obs["celltype"] == ct
        n_cells = mask.sum()
        donors_in_ct = adata.obs.loc[mask, "donor"].unique()
        if n_cells < 20 or len(donors_in_ct) < 2:
            continue
        labels = adata.obs.loc[mask, "donor"].values
        try:
            scores = silhouette_samples(pca_coords[mask], labels, metric="euclidean")
            results.append({"celltype": ct, "mean_silhouette": scores.mean(),
                            "n_cells": n_cells, "n_donors": len(donors_in_ct)})
        except Exception:
            pass

    if not results:
        print("  Silhouette: not enough cells/donors for any cell type.")
        return

    df = pd.DataFrame(results).sort_values("mean_silhouette", ascending=False)

    fig, ax = plt.subplots(figsize=(8, 4))
    colors = [("#E07B54" if s > 0.2 else "#5B8DB8") for s in df["mean_silhouette"]]
    ax.barh(df["celltype"], df["mean_silhouette"], color=colors)
    ax.axvline(0.2, color="red", ls="--", lw=1, label="Threshold 0.2")
    ax.set_xlabel("Mean silhouette score (donor labels, PCA space)", fontsize=10)
    ax.set_title("Donor-mixing per cell type\n"
                 "(high score = donor separation → may need batch correction)",
                 fontsize=10)
    ax.legend(fontsize=9)
    plt.tight_layout()
    fig.savefig(outpath, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {outpath}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Summary tables
# ─────────────────────────────────────────────────────────────────────────────

def save_nuclei_table(adata, outpath):
    tbl = (adata.obs.groupby(["donor", "celltype"])
           .size()
           .unstack(fill_value=0))
    tbl.to_csv(outpath)
    print(f"  Saved: {outpath}")
    return tbl


def save_cluster_label_concordance(adata, outpath):
    """Fraction of each Leiden cluster belonging to each provided cell-type label."""
    ct_col = "celltype"
    cross   = pd.crosstab(adata.obs["leiden"], adata.obs[ct_col])
    frac    = cross.div(cross.sum(axis=1), axis=0).round(3)
    frac.to_csv(outpath)
    print(f"  Saved: {outpath}")
    return frac


# ─────────────────────────────────────────────────────────────────────────────
# SCTransform-style HVG comparison (secondary check)
# ─────────────────────────────────────────────────────────────────────────────

def compare_hvg_methods(adata_raw_counts, outdir):
    """
    Compare top-2000 HVGs selected by log-normalisation vs. Pearson-residual
    (SCTransform proxy via scanpy) and save a Venn-style summary CSV.
    """
    try:
        adata2 = ad.AnnData(X=adata_raw_counts.layers["counts"].copy(),
                             obs=adata_raw_counts.obs.copy(),
                             var=adata_raw_counts.var.copy())
        sc.pp.normalize_total(adata2, target_sum=1e4)
        sc.pp.log1p(adata2)
        sc.pp.highly_variable_genes(adata2, n_top_genes=N_HVG, flavor="seurat")
        hvg_lognorm = set(adata2.var_names[adata2.var["highly_variable"]])

        adata3 = ad.AnnData(X=adata_raw_counts.layers["counts"].copy(),
                             obs=adata_raw_counts.obs.copy(),
                             var=adata_raw_counts.var.copy())
        sc.experimental.pp.normalize_pearson_residuals(adata3)
        sc.pp.highly_variable_genes(adata3, n_top_genes=N_HVG,
                                    flavor="seurat_v3")
        hvg_sct = set(adata3.var_names[adata3.var["highly_variable"]])

        overlap  = len(hvg_lognorm & hvg_sct)
        summary  = pd.DataFrame([{
            "method_A"      : "log-norm",
            "method_B"      : "SCTransform-proxy",
            "HVGs_A"        : len(hvg_lognorm),
            "HVGs_B"        : len(hvg_sct),
            "overlap"       : overlap,
            "jaccard_index" : round(overlap / len(hvg_lognorm | hvg_sct), 3),
        }])
        out = os.path.join(outdir, "hvg_method_comparison.csv")
        summary.to_csv(out, index=False)
        print(f"  HVG method comparison saved: {out}")
        print(f"    log-norm ∩ SCT-proxy: {overlap}/{N_HVG} genes "
              f"(Jaccard={summary['jaccard_index'].iloc[0]})")
    except Exception as e:
        print(f"  SCTransform-proxy comparison skipped: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="BST281 Analysis 2: Normalisation, embedding, annotation validation"
    )
    p.add_argument("--adata",   required=True,
                   help="Path to adata_filtered.h5ad from Analysis 1")
    p.add_argument("--outdir",  default="results/analysis2")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    print("=" * 60)
    print("Analysis 2 — Normalisation, Embedding, Annotation Validation")
    print(f"  Seed: {SEED}")
    print("=" * 60)

    # ── 1. Load filtered AnnData ─────────────────────────────────────────────
    print("\n[1/7] Loading filtered AnnData ...")
    adata = ad.read_h5ad(args.adata)
    adata = harmonise_obs(adata)
    print(f"  Shape: {adata.shape[0]:,} nuclei × {adata.shape[1]:,} genes")
    print(f"  Cell types: {sorted(adata.obs['celltype'].unique())}")
    print(f"  Donors    : {sorted(adata.obs['donor'].unique())}")

    # ── 2. Preprocessing & embedding ────────────────────────────────────────
    print("\n[2/7] Normalisation, HVG selection, PCA, UMAP, clustering ...")
    adata = run_preprocessing(adata)

    # ── 3. HVG method comparison (secondary check) ──────────────────────────
    print("\n[3/7] Comparing HVG methods (log-norm vs SCTransform-proxy) ...")
    compare_hvg_methods(adata, args.outdir)

    # ── 4. UMAP visualisations ───────────────────────────────────────────────
    print("\n[4/7] Generating UMAP plots ...")
    plot_umaps(adata, args.outdir)

    # ── 5. Marker gene validation ────────────────────────────────────────────
    print("\n[5/7] Marker gene validation plots ...")
    marker_dict = filter_markers_present(adata, MARKER_GENES)

    if marker_dict:
        plot_marker_dotplot(adata, marker_dict,
                            os.path.join(args.outdir, "fig_markers_dotplot.png"))
        plot_marker_violin(adata, marker_dict,
                           os.path.join(args.outdir, "fig_markers_violin.png"))
        plot_marker_heatmap(adata, marker_dict,
                            os.path.join(args.outdir, "fig_marker_heatmap.png"))
    else:
        print("  No canonical markers found in dataset — skipping marker plots.")

    # ── 6. Quantitative summary tables ──────────────────────────────────────
    print("\n[6/7] Saving summary tables ...")
    save_nuclei_table(adata,
                      os.path.join(args.outdir,
                                   "table_nuclei_per_donor_celltype.csv"))
    save_cluster_label_concordance(
        adata,
        os.path.join(args.outdir, "table_cluster_label_concordance.csv")
    )

    # ── 7. Donor-mixing silhouette ───────────────────────────────────────────
    print("\n[7/7] Computing donor-mixing silhouette scores ...")
    sil_df = compute_donor_mixing(
        adata,
        os.path.join(args.outdir, "fig_donor_mixing.png")
    )
    if sil_df is not None:
        sil_df.to_csv(os.path.join(args.outdir,
                                   "table_donor_silhouette.csv"), index=False)
        high_sil = sil_df[sil_df["mean_silhouette"] > 0.2]
        if not high_sil.empty:
            print(f"  ⚠ Cell types with silhouette > 0.2 (possible batch effect):")
            print(high_sil[["celltype", "mean_silhouette"]].to_string(index=False))
        else:
            print("  ✓ No cell type shows strong donor-specific separation (sil ≤ 0.2).")

    # ── Save final AnnData ───────────────────────────────────────────────────
    h5ad_path = os.path.join(args.outdir, "adata_embedded.h5ad")
    adata.write_h5ad(h5ad_path)
    print(f"\n  Final AnnData saved: {h5ad_path}")

    print("\n✓ Analysis 2 complete.")
    print(f"  All outputs in: {os.path.abspath(args.outdir)}")
    print("\n  Deliverable figures:")
    for f in sorted(os.listdir(args.outdir)):
        if f.endswith(".png") or f.endswith(".csv"):
            print(f"    {f}")


if __name__ == "__main__":
    main()
