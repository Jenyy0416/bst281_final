"""
BST 281 Final Project — Module 1, Analysis 1
Data loading, QC filtering, and donor-level quantitative checks

Author  : Jenny Yang
Dataset : Zenodo record 17302976
  - adsn_matrix.mtx.gz   (sparse count matrix)
  - adsn_barcodes.txt    (cell barcodes, if separate)
  - adsn_features.txt    (gene names,    if separate)
  - adsn_metadata.txt    (cell-level metadata)

Usage:
    python analysis1_qc.py --matrix adsn_matrix.mtx.gz \
                           --features adsn_features.txt \
                           --barcodes adsn_barcodes.txt \
                           --metadata adsn_metadata.txt \
                           --outdir results/analysis1

Outputs (all saved to --outdir):
    qc_summary_table.csv          - per-donor QC summary (median metrics, nuclei counts)
    nuclei_before_after.csv       - nuclei retained before / after filtering per donor
    fig_qc_violin.png             - per-donor violin plots of nFeature, nCount, pctMT
    fig_qc_hist.png               - distribution histograms of QC metrics
    fig_nuclei_bar.png            - nuclei per donor coloured by disease group
    qc_imbalance_test.csv         - chi-square / Wilcoxon test results for donor imbalance
    adata_filtered.h5ad           - QC-filtered AnnData object for Analysis 2
"""

import argparse
import os
import random

import numpy as np
import pandas as pd
import scipy.io
import scipy.sparse
import scipy.stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import anndata as ad

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ── QC thresholds (from revised proposal) ────────────────────────────────────
MIN_GENES   = 200
MAX_GENES   = 6_000
MAX_COUNTS  = 20_000
MAX_PCT_MT  = 5.0        # percent

# ── Plotting defaults ─────────────────────────────────────────────────────────
PALETTE = sns.color_palette("Set2")
FIG_DPI = 150


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_matrix_market(matrix_path, features_path, barcodes_path):
    """Load a Matrix Market sparse matrix with matching features / barcodes."""
    X = scipy.io.mmread(matrix_path).T.tocsr()   # cells × genes
    genes    = pd.read_csv(features_path, header=None, sep="\t")[0].tolist()
    barcodes = pd.read_csv(barcodes_path, header=None, sep="\t")[0].tolist()
    return X, genes, barcodes


def load_metadata(metadata_path):
    """
    Load cell-level metadata.
    Tries tab-separated first, then comma-separated.
    """
    for sep in ("\t", ","):
        try:
            df = pd.read_csv(metadata_path, sep=sep, index_col=0)
            if df.shape[1] > 1:
                return df
        except Exception:
            pass
    raise ValueError(f"Cannot parse metadata at {metadata_path}")


def harmonise_columns(meta):
    """
    Normalise common column name variants so the rest of the script
    uses consistent names: nFeature_RNA, nCount_RNA, pctMT, donor, disease.

    For this Zenodo dataset, disease is derived from batchCond or subclustCond
    (e.g. "AD_batch1" -> "AD", "Control_batch2" -> "Control").
    """
    rename = {}
    col_lower = {c.lower(): c for c in meta.columns}

    candidates = {
        "nFeature_RNA": ["nfeature_rna", "n_genes", "ngenes", "n_genes_by_counts"],
        "nCount_RNA"  : ["ncount_rna",  "n_counts", "ncounts", "total_counts"],
        "pctMT"       : ["pctmt", "pct_mt", "percent_mt", "pct_counts_mt",
                         "percent.mt", "pct.mt"],
        "donor"       : ["donor", "sample", "donor_id", "sampleid", "patient"],
        "disease"     : ["disease", "diagnosis", "group", "condition",
                         "disease_group", "ad_ctrl"],
    }
    for target, options in candidates.items():
        for opt in options:
            if opt in col_lower and target not in meta.columns:
                rename[col_lower[opt]] = target
    meta = meta.rename(columns=rename)

    # If disease still missing, derive it from batchCond or subclustCond
    # Expected format: "AD_batch1", "Control_batch2", "AD_Microglia", etc.
    if "disease" not in meta.columns:
        for src_col in ["batchCond", "subclustCond", "batchcond", "subclustcond"]:
            if src_col in meta.columns:
                def _extract_disease(val):
                    val = str(val)
                    if val.lower().startswith("ad"):
                        return "AD"
                    elif val.lower().startswith("control") or val.lower().startswith("ctrl"):
                        return "Control"
                    # Fallback: take everything before first underscore
                    return val.split("_")[0]
                meta["disease"] = meta[src_col].apply(_extract_disease)
                print(f"  Derived 'disease' column from '{src_col}': "
                      f"{sorted(meta['disease'].unique())}")
                break
        if "disease" not in meta.columns:
            print("  Warning: could not derive 'disease' column. "
                  "Setting all cells to 'Unknown'.")
            meta["disease"] = "Unknown"
    return meta


def compute_qc_if_missing(adata):
    """
    If nFeature_RNA / nCount_RNA / pctMT are missing from obs,
    compute them from the count matrix.
    """
    obs = adata.obs
    if "nFeature_RNA" not in obs.columns:
        obs["nFeature_RNA"] = np.diff(adata.X.tocsc().indptr) if hasattr(adata.X, "indptr") \
            else (adata.X > 0).sum(axis=1).A1
        # safer: count non-zero per row
        obs["nFeature_RNA"] = np.array((adata.X > 0).sum(axis=1)).flatten()
    if "nCount_RNA" not in obs.columns:
        obs["nCount_RNA"] = np.array(adata.X.sum(axis=1)).flatten()
    if "pctMT" not in obs.columns:
        mt_genes = [g for g in adata.var_names if g.upper().startswith("MT-")]
        if mt_genes:
            mt_idx = adata.var_names.isin(mt_genes)
            mt_counts = np.array(adata.X[:, mt_idx].sum(axis=1)).flatten()
            obs["pctMT"] = mt_counts / obs["nCount_RNA"].clip(lower=1) * 100
        else:
            obs["pctMT"] = 0.0
            print("  Warning: no MT- genes found; pctMT set to 0 for all cells.")
    adata.obs = obs
    return adata


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 — Per-donor violin plots of QC metrics
# ─────────────────────────────────────────────────────────────────────────────

def plot_qc_violin(adata, outpath):
    metrics   = ["nFeature_RNA", "nCount_RNA", "pctMT"]
    labels    = ["Genes / nucleus", "UMI counts / nucleus", "% MT reads"]
    thresholds = [
        (MIN_GENES,  "min", "blue"),
        (MAX_GENES,  "max", "red"),
        (MAX_COUNTS, "max", "red"),
        (MAX_PCT_MT, "max", "red"),
    ]

    donors = sorted(adata.obs["donor"].unique())
    n_donors = len(donors)

    fig, axes = plt.subplots(1, 3, figsize=(max(10, n_donors * 0.9 + 2), 5))
    fig.suptitle("QC metric distributions per donor (pre-filter)", fontsize=13, y=1.02)

    for ax, metric, label in zip(axes, metrics, labels):
        data_list = [adata.obs.loc[adata.obs["donor"] == d, metric].values for d in donors]
        parts = ax.violinplot(data_list, positions=range(n_donors),
                              showmedians=True, showextrema=True)
        for pc in parts["bodies"]:
            pc.set_facecolor(PALETTE[2])
            pc.set_alpha(0.7)

        # threshold lines
        if metric == "nFeature_RNA":
            ax.axhline(MIN_GENES, color="blue",  ls="--", lw=1, label=f"min={MIN_GENES}")
            ax.axhline(MAX_GENES, color="red",   ls="--", lw=1, label=f"max={MAX_GENES}")
        elif metric == "nCount_RNA":
            ax.axhline(MAX_COUNTS, color="red",  ls="--", lw=1, label=f"max={MAX_COUNTS}")
        else:
            ax.axhline(MAX_PCT_MT, color="red",  ls="--", lw=1, label=f"max={MAX_PCT_MT}%")

        ax.set_xticks(range(n_donors))
        ax.set_xticklabels(donors, rotation=60, ha="right", fontsize=7)
        ax.set_ylabel(label, fontsize=9)
        ax.set_title(metric, fontsize=10)
        ax.legend(fontsize=7)

    plt.tight_layout()
    fig.savefig(outpath, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {outpath}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 — Overall histogram distributions
# ─────────────────────────────────────────────────────────────────────────────

def plot_qc_histograms(adata, outpath):
    metrics    = ["nFeature_RNA", "nCount_RNA", "pctMT"]
    xlabels    = ["Genes / nucleus", "UMI counts / nucleus", "% MT reads"]
    vlines     = [(MIN_GENES, "blue"), (MAX_GENES, "red"),
                  (MAX_COUNTS, "red"), (MAX_PCT_MT, "red")]
    per_metric = {
        "nFeature_RNA": [(MIN_GENES, "blue", f"min={MIN_GENES}"),
                         (MAX_GENES, "red",  f"max={MAX_GENES}")],
        "nCount_RNA"  : [(MAX_COUNTS, "red", f"max={MAX_COUNTS}")],
        "pctMT"       : [(MAX_PCT_MT, "red", f"max={MAX_PCT_MT}%")],
    }

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle("Overall QC metric distributions (pre-filter)", fontsize=12)

    for ax, metric, xlabel in zip(axes, metrics, xlabels):
        vals = adata.obs[metric].values
        ax.hist(vals, bins=80, color=PALETTE[0], edgecolor="none", alpha=0.8)
        for vline, color, lbl in per_metric[metric]:
            ax.axvline(vline, color=color, ls="--", lw=1.5, label=lbl)
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel("# nuclei", fontsize=9)
        ax.set_title(metric, fontsize=10)
        ax.legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(outpath, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {outpath}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3 — Nuclei per donor, coloured by disease group
# ─────────────────────────────────────────────────────────────────────────────

def plot_nuclei_bar(before_df, after_df, outpath):
    """
    Stacked-style bar chart: before and after filtering,
    bars coloured by disease group.
    """
    # Merge
    merged = before_df[["donor", "disease", "n_before"]].merge(
        after_df[["donor", "n_after"]], on="donor", how="left"
    ).fillna(0)
    merged = merged.sort_values(["disease", "donor"])

    donors  = merged["donor"].tolist()
    disease = merged["disease"].tolist()
    n_before = merged["n_before"].values
    n_after  = merged["n_after"].values.astype(int)

    disease_cats = sorted(set(disease))
    color_map    = {d: PALETTE[i] for i, d in enumerate(disease_cats)}
    bar_colors   = [color_map[d] for d in disease]

    x = np.arange(len(donors))
    width = 0.38

    fig, ax = plt.subplots(figsize=(max(8, len(donors) * 0.65), 5))
    bars_b = ax.bar(x - width / 2, n_before, width, color=bar_colors, alpha=0.4,
                    edgecolor="grey", linewidth=0.5, label="Before filter")
    bars_a = ax.bar(x + width / 2, n_after,  width, color=bar_colors, alpha=0.9,
                    edgecolor="grey", linewidth=0.5, label="After filter")

    # Legend for disease groups
    from matplotlib.patches import Patch
    legend_patches = [Patch(facecolor=color_map[d], label=d) for d in disease_cats]
    leg1 = ax.legend(handles=legend_patches, title="Disease", loc="upper right", fontsize=8)
    ax.add_artist(leg1)
    # Hatch legend
    from matplotlib.patches import Patch as P2
    ax.legend(handles=[P2(facecolor="grey", alpha=0.4, label="Before filter"),
                        P2(facecolor="grey", alpha=0.9, label="After filter")],
              loc="upper left", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(donors, rotation=60, ha="right", fontsize=8)
    ax.set_ylabel("# nuclei", fontsize=10)
    ax.set_title("Nuclei per donor — before vs. after QC filtering", fontsize=11)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
    plt.tight_layout()
    fig.savefig(outpath, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {outpath}")


# ─────────────────────────────────────────────────────────────────────────────
# Statistical tests — donor imbalance
# ─────────────────────────────────────────────────────────────────────────────

def run_imbalance_tests(adata_pre, adata_post, meta_pre):
    """
    1. Chi-square: is nuclei attrition (removed/kept) independent of disease?
    2. Wilcoxon rank-sum: do AD and ctrl donors differ in median QC metrics pre-filter?
    Returns a DataFrame of results.
    """
    results = []

    # ── 1. Chi-square on attrition ──────────────────────────────────────────
    donors_pre  = meta_pre.groupby("donor")["disease"].first()
    n_pre_map   = meta_pre.groupby("donor").size().rename("n_pre")
    n_post_map  = adata_post.obs.groupby("donor").size().rename("n_post")
    counts      = pd.concat([donors_pre, n_pre_map, n_post_map], axis=1).fillna(0)
    counts["n_removed"] = counts["n_pre"] - counts["n_post"]

    disease_cats = counts["disease"].unique()
    if len(disease_cats) >= 2:
        ctable = counts.groupby("disease")[["n_post", "n_removed"]].sum()
        chi2, p_chi, dof, _ = scipy.stats.chi2_contingency(ctable.values)
        results.append({
            "test"       : "Chi-square (attrition vs disease)",
            "statistic"  : round(chi2, 4),
            "p_value"    : round(p_chi, 6),
            "dof"        : dof,
            "note"       : f"Contingency: {ctable.to_dict()}",
        })

    # ── 2. Wilcoxon on QC metrics ───────────────────────────────────────────
    donor_medians = (meta_pre
                     .groupby("donor")[["nFeature_RNA", "nCount_RNA", "pctMT"]]
                     .median())
    donor_disease = meta_pre.groupby("donor")["disease"].first()
    donor_medians = donor_medians.join(donor_disease)

    if len(disease_cats) >= 2:
        grp0 = disease_cats[0]
        grp1 = disease_cats[1]
        for metric in ["nFeature_RNA", "nCount_RNA", "pctMT"]:
            a = donor_medians.loc[donor_medians["disease"] == grp0, metric].dropna()
            b = donor_medians.loc[donor_medians["disease"] == grp1, metric].dropna()
            if len(a) >= 2 and len(b) >= 2:
                stat, pval = scipy.stats.mannwhitneyu(a, b, alternative="two-sided")
                results.append({
                    "test"      : f"Wilcoxon ({metric}: {grp0} vs {grp1})",
                    "statistic" : round(stat, 4),
                    "p_value"   : round(pval, 6),
                    "dof"       : "—",
                    "note"      : f"n={len(a)}, {len(b)}; medians={a.median():.1f}, {b.median():.1f}",
                })

    return pd.DataFrame(results)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="BST281 Analysis 1: QC and filtering")
    p.add_argument("--matrix",   required=True,  help="Path to adsn_matrix.mtx.gz")
    p.add_argument("--features", default=None,   help="Path to features/genes file (optional if embedded in mtx dir)")
    p.add_argument("--barcodes", default=None,   help="Path to barcodes file (optional)")
    p.add_argument("--metadata", required=True,  help="Path to adsn_metadata.txt")
    p.add_argument("--outdir",   default="results/analysis1")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    print("=" * 60)
    print("Analysis 1 — QC filtering and donor-level checks")
    print(f"  Seed: {SEED}")
    print("=" * 60)

    # ── 1. Load metadata ─────────────────────────────────────────────────────
    print("\n[1/6] Loading metadata ...")
    meta = load_metadata(args.metadata)
    meta = harmonise_columns(meta)
    print(f"  Metadata shape: {meta.shape}")
    print(f"  Columns detected: {meta.columns.tolist()}")

    # ── 2. Load count matrix ─────────────────────────────────────────────────
    print("\n[2/6] Loading count matrix ...")

    # Resolve features / barcodes paths
    # Priority: explicit CLI arg (trusted as-is) -> auto-detect next to matrix
    matrix_dir = os.path.dirname(os.path.abspath(args.matrix))

    def _find(candidates, explicit):
        if explicit:
            # Trust the explicit path; try as-is first, then relative to matrix dir
            if os.path.exists(explicit):
                return os.path.abspath(explicit)
            alt = os.path.join(matrix_dir, os.path.basename(explicit))
            if os.path.exists(alt):
                return alt
            # Return as-is even if not verified; pandas/scipy will give clear error
            return explicit
        for cand in candidates:
            path = os.path.join(matrix_dir, cand)
            if os.path.exists(path):
                return path
        return None

    features_path = _find(
        ["adsn_features.tsv.gz", "features.tsv.gz", "features.tsv",
         "genes.tsv.gz", "genes.tsv", "adsn_features.txt"],
        args.features
    )
    barcodes_path = _find(
        ["adsn_barcodes.tsv.gz", "barcodes.tsv.gz",
         "barcodes.tsv", "adsn_barcodes.txt"],
        args.barcodes
    )

    if features_path and barcodes_path:
        X, genes, barcodes = load_matrix_market(args.matrix, features_path, barcodes_path)
        adata = ad.AnnData(X=X)
        adata.obs_names = barcodes
        adata.var_names = genes
    else:
        # Fallback: try reading the mtx directory directly with scanpy
        try:
            import scanpy as sc
            adata = sc.read_10x_mtx(matrix_dir, var_names="gene_symbols",
                                     cache=False, gex_only=False)
        except Exception as e:
            raise RuntimeError(
                f"Could not load matrix: {e}\n"
                "Please supply --features and --barcodes explicitly."
            )

    print(f"  Matrix loaded: {adata.shape[0]:,} nuclei × {adata.shape[1]:,} genes")

    # ── 3. Attach metadata ───────────────────────────────────────────────────
    print("\n[3/6] Attaching metadata ...")
    # Align on barcode index
    common = adata.obs_names.intersection(meta.index)
    if len(common) == 0:
        # Try stripping suffixes like "-1"
        meta_stripped = meta.copy()
        meta_stripped.index = meta_stripped.index.str.replace(r"-\d+$", "", regex=True)
        common = adata.obs_names.intersection(meta_stripped.index)
        if len(common) > 0:
            meta = meta_stripped
    if len(common) == 0:
        print("  Warning: barcodes in matrix and metadata do not overlap — "
              "attaching metadata by row position.")
        n = min(len(adata), len(meta))
        adata = adata[:n]
        adata.obs = meta.iloc[:n].set_index(adata.obs_names)
    else:
        adata = adata[common]
        adata.obs = meta.loc[common]
    print(f"  Nuclei after barcode alignment: {adata.n_obs:,}")

    # ── 4. Compute / verify QC metrics ──────────────────────────────────────
    print("\n[4/6] Computing QC metrics ...")
    adata = compute_qc_if_missing(adata)

    # Record pre-filter state
    meta_pre = adata.obs.copy()
    n_pre    = len(adata)
    before_df = (meta_pre.groupby("donor")
                 .agg(n_before=("nFeature_RNA", "count"),
                      disease=("disease", "first"))
                 .reset_index()
                 if "disease" in meta_pre.columns
                 else meta_pre.groupby("donor")
                 .agg(n_before=("nFeature_RNA", "count"))
                 .reset_index()
                 .assign(disease="Unknown"))

    # ── 5. QC visualisations (pre-filter) ───────────────────────────────────
    print("\n[5/6] Generating pre-filter QC plots ...")
    plot_qc_violin(adata,
                   os.path.join(args.outdir, "fig_qc_violin.png"))
    plot_qc_histograms(adata,
                       os.path.join(args.outdir, "fig_qc_hist.png"))

    # ── 6. Apply filters ─────────────────────────────────────────────────────
    print("\n[6/6] Applying QC filters ...")
    keep = (
        (adata.obs["nFeature_RNA"] >= MIN_GENES)  &
        (adata.obs["nFeature_RNA"] <= MAX_GENES)  &
        (adata.obs["nCount_RNA"]   <= MAX_COUNTS) &
        (adata.obs["pctMT"]        <= MAX_PCT_MT)
    )

    # If doublet column exists, also filter predicted doublets
    doublet_col = next(
        (c for c in adata.obs.columns
         if "doublet" in c.lower() and "score" not in c.lower()), None
    )
    if doublet_col:
        doublet_mask = adata.obs[doublet_col].astype(str).str.lower().isin(
            ["true", "1", "doublet"]
        )
        n_doublets = doublet_mask.sum()
        keep = keep & ~doublet_mask
        print(f"  Doublet filter ({doublet_col}): removed {n_doublets:,} predicted doublets")

    adata_filt = adata[keep].copy()
    n_post = len(adata_filt)
    print(f"  Nuclei before filter : {n_pre:,}")
    print(f"  Nuclei after  filter : {n_post:,}  "
          f"({100 * n_post / n_pre:.1f}% retained)")

    after_df = (adata_filt.obs.groupby("donor")
                .agg(n_after=("nFeature_RNA", "count"))
                .reset_index())

    # ── QC summary table ─────────────────────────────────────────────────────
    summary = (meta_pre
               .groupby("donor")
               .agg(
                   disease      = ("disease", "first"),
                   n_nuclei_pre = ("nFeature_RNA", "count"),
                   median_genes = ("nFeature_RNA", "median"),
                   median_umi   = ("nCount_RNA",   "median"),
                   median_pctMT = ("pctMT",        "median"),
               )
               .reset_index())
    summary = summary.merge(after_df.rename(columns={"n_after": "n_nuclei_post"}),
                            on="donor", how="left")
    summary["pct_retained"] = (summary["n_nuclei_post"] / summary["n_nuclei_pre"] * 100).round(1)
    summary.to_csv(os.path.join(args.outdir, "qc_summary_table.csv"), index=False)
    print(f"\n  QC summary table:\n{summary.to_string(index=False)}")

    before_after = summary[["donor", "disease", "n_nuclei_pre",
                              "n_nuclei_post", "pct_retained"]]
    before_after.to_csv(os.path.join(args.outdir, "nuclei_before_after.csv"), index=False)

    # ── Nuclei bar chart ──────────────────────────────────────────────────────
    plot_nuclei_bar(before_df, after_df,
                    os.path.join(args.outdir, "fig_nuclei_bar.png"))

    # ── Statistical imbalance tests ───────────────────────────────────────────
    test_results = run_imbalance_tests(adata_filt, adata_filt, meta_pre)
    test_results.to_csv(os.path.join(args.outdir, "qc_imbalance_test.csv"), index=False)
    print(f"\n  Imbalance test results:\n{test_results.to_string(index=False)}")

    # ── Save filtered AnnData ─────────────────────────────────────────────────
    h5ad_path = os.path.join(args.outdir, "adata_filtered.h5ad")
    adata_filt.write_h5ad(h5ad_path)
    print(f"\n  Filtered AnnData saved: {h5ad_path}")

    print("\n✓ Analysis 1 complete.")
    print(f"  All outputs in: {os.path.abspath(args.outdir)}")


if __name__ == "__main__":
    main()
