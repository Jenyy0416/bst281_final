"""
Module 2 Analysis 2: true within-cell-type subclustering

Purpose:
- For AD-relevant focal cell types, subset each cell type and rerun Leiden clustering.
- Calculate donor-level subcluster proportions within each focal cell type.
- Apply low-count filters before exploratory statistical testing.

This is more defensible than using the global Leiden labels as "subclusters".

Run from the project root:
    python module2_analysis2_true_subclustering.py
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests


# ============================================================
# Paths and configuration
# ============================================================

project_dir = Path(__file__).resolve().parent
outdir = project_dir / "results/module2_analysis2_subclustering"
outdir.mkdir(parents=True, exist_ok=True)

data_results_dir = project_dir / "data/results"
h5ad_path = data_results_dir / "adata_embedded_fixed.h5ad"

UNCLEAR_DONORS = ["ad-un", "ct-un"]
FOCUS_CELLTYPES = ["neuron", "astro", "mg"]

# Subclustering settings
LEIDEN_RESOLUTION = 0.3
MAX_N_PCS = 30
MAX_N_NEIGHBORS = 15

# Low-count filters for statistical tests
MIN_TOTAL_NUCLEI_FOR_CELLTYPE = 80
MIN_NUCLEI_PER_DONOR_CELLTYPE = 5
MIN_TOTAL_NUCLEI_PER_SUBCLUSTER = 20
MIN_DONORS_PRESENT_PER_SUBCLUSTER = 3
MIN_DONORS_PRESENT_PER_GROUP = 1


# ============================================================
# Helper functions
# ============================================================

def safe_int_sort(values):
    def key_func(x):
        try:
            return int(x)
        except Exception:
            return str(x)
    return sorted(values, key=key_func)


def get_existing_hvg_subset(adata_ct):
    """Use existing highly variable genes if available and enough are retained."""
    if "highly_variable" in adata_ct.var.columns:
        hvg = adata_ct.var["highly_variable"].astype(bool).values
        if hvg.sum() >= 50:
            return adata_ct[:, hvg].copy()
    return adata_ct.copy()


def run_subclustering(adata_ct, celltype):
    """Run PCA, neighbors, Leiden, and UMAP on one focal cell type."""
    adata_sub = get_existing_hvg_subset(adata_ct)

    n_obs = adata_sub.n_obs
    n_vars = adata_sub.n_vars

    if n_obs < MIN_TOTAL_NUCLEI_FOR_CELLTYPE:
        print(f"Skipping {celltype}: only {n_obs} nuclei.")
        return None

    n_comps = min(MAX_N_PCS, n_obs - 1, n_vars - 1)
    n_neighbors = min(MAX_N_NEIGHBORS, n_obs - 1)

    if n_comps < 2 or n_neighbors < 2:
        print(f"Skipping {celltype}: not enough observations or variables for PCA/neighbors.")
        return None

    # The input object is assumed to come from Module 1 preprocessing.
    # We rerun dimensionality reduction and graph clustering within each cell type.
    sc.pp.pca(adata_sub, n_comps=n_comps)
    sc.pp.neighbors(adata_sub, n_neighbors=n_neighbors, n_pcs=n_comps)
    sc.tl.leiden(
        adata_sub,
        resolution=LEIDEN_RESOLUTION,
        key_added="subcluster"
    )
    sc.tl.umap(adata_sub)

    # Copy subcluster labels back to the full-gene cell-type object.
    adata_ct = adata_ct.copy()
    adata_ct.obs["subcluster"] = adata_sub.obs["subcluster"].astype(str)
    adata_ct.obsm["X_umap_subcluster"] = adata_sub.obsm["X_umap"]

    return adata_ct


def save_umap_plots(adata_ct, celltype):
    """Save simple UMAP plots for subclusters and disease labels."""
    coords = adata_ct.obsm["X_umap_subcluster"]
    plot_df = pd.DataFrame({
        "UMAP1": coords[:, 0],
        "UMAP2": coords[:, 1],
        "subcluster": adata_ct.obs["subcluster"].astype(str).values,
        "disease": adata_ct.obs["disease"].astype(str).values,
        "donor": adata_ct.obs["donor"].astype(str).values
    })

    for color_col in ["subcluster", "disease"]:
        plt.figure(figsize=(6, 5))
        sns.scatterplot(
            data=plot_df,
            x="UMAP1",
            y="UMAP2",
            hue=color_col,
            s=12,
            linewidth=0,
            alpha=0.8
        )
        plt.title(f"{celltype}: within-cell-type subclustering colored by {color_col}")
        plt.xlabel("UMAP1")
        plt.ylabel("UMAP2")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", title=color_col)
        plt.tight_layout()
        plt.savefig(outdir / f"module2_{celltype}_true_subcluster_umap_by_{color_col}.png", dpi=300)
        plt.close()


def count_subclusters(adata_ct, celltype):
    """Create zero-filled donor-level subcluster proportion table."""
    obs_ct = adata_ct.obs[["donor", "disease", "subcluster"]].copy()
    obs_ct["donor"] = obs_ct["donor"].astype(str)
    obs_ct["disease"] = obs_ct["disease"].astype(str)
    obs_ct["subcluster"] = obs_ct["subcluster"].astype(str)
    obs_ct["celltype"] = celltype

    donor_celltype_counts = (
        obs_ct
        .groupby(["donor", "disease"])
        .size()
        .reset_index(name="total_nuclei_per_donor_celltype")
    )

    donor_celltype_counts = donor_celltype_counts[
        donor_celltype_counts["total_nuclei_per_donor_celltype"] >= MIN_NUCLEI_PER_DONOR_CELLTYPE
    ].copy()

    valid_donors = donor_celltype_counts["donor"].tolist()
    obs_ct = obs_ct[obs_ct["donor"].isin(valid_donors)].copy()

    observed_counts = (
        obs_ct
        .groupby(["donor", "disease", "celltype", "subcluster"])
        .size()
        .reset_index(name="n_nuclei")
    )

    donors_ct = donor_celltype_counts[["donor", "disease"]].drop_duplicates()
    subclusters_ct = safe_int_sort(obs_ct["subcluster"].unique())

    all_combos = pd.MultiIndex.from_product(
        [donors_ct["donor"], [celltype], subclusters_ct],
        names=["donor", "celltype", "subcluster"]
    ).to_frame(index=False)

    all_combos = all_combos.merge(donors_ct, on="donor", how="left")

    sub_counts = all_combos.merge(
        observed_counts,
        on=["donor", "disease", "celltype", "subcluster"],
        how="left"
    )
    sub_counts["n_nuclei"] = sub_counts["n_nuclei"].fillna(0).astype(int)

    sub_counts = sub_counts.merge(
        donor_celltype_counts,
        on=["donor", "disease"],
        how="left"
    )

    sub_counts["subcluster_proportion"] = (
        sub_counts["n_nuclei"] / sub_counts["total_nuclei_per_donor_celltype"]
    )

    return sub_counts


def summarize_subclusters(sub_counts):
    """Create cluster-level count and prevalence summary for low-count filtering."""
    present = sub_counts[sub_counts["n_nuclei"] > 0].copy()

    total_summary = (
        sub_counts
        .groupby(["celltype", "subcluster"])
        .agg(
            total_nuclei=("n_nuclei", "sum"),
            n_donors_tested=("donor", "nunique"),
            mean_proportion_all=("subcluster_proportion", "mean"),
            median_proportion_all=("subcluster_proportion", "median")
        )
        .reset_index()
    )

    present_summary = (
        present
        .groupby(["celltype", "subcluster"])
        .agg(
            n_donors_present=("donor", "nunique")
        )
        .reset_index()
    )

    group_present = (
        present
        .groupby(["celltype", "subcluster", "disease"])
        .agg(n_donors_present_group=("donor", "nunique"))
        .reset_index()
        .pivot_table(
            index=["celltype", "subcluster"],
            columns="disease",
            values="n_donors_present_group",
            fill_value=0
        )
        .reset_index()
    )

    group_means = (
        sub_counts
        .groupby(["celltype", "subcluster", "disease"])
        .agg(mean_proportion=("subcluster_proportion", "mean"))
        .reset_index()
        .pivot_table(
            index=["celltype", "subcluster"],
            columns="disease",
            values="mean_proportion",
            fill_value=np.nan
        )
        .reset_index()
    )

    # Rename disease-derived columns safely.
    group_present = group_present.rename(columns={
        "AD": "n_AD_donors_present",
        "Control": "n_Control_donors_present"
    })
    group_means = group_means.rename(columns={
        "AD": "mean_proportion_AD",
        "Control": "mean_proportion_Control"
    })

    summary = total_summary.merge(present_summary, on=["celltype", "subcluster"], how="left")
    summary = summary.merge(group_present, on=["celltype", "subcluster"], how="left")
    summary = summary.merge(group_means, on=["celltype", "subcluster"], how="left")

    for c in ["n_donors_present", "n_AD_donors_present", "n_Control_donors_present"]:
        if c not in summary.columns:
            summary[c] = 0
        summary[c] = summary[c].fillna(0).astype(int)

    summary["passes_low_count_filter"] = (
        (summary["total_nuclei"] >= MIN_TOTAL_NUCLEI_PER_SUBCLUSTER) &
        (summary["n_donors_present"] >= MIN_DONORS_PRESENT_PER_SUBCLUSTER) &
        (summary["n_AD_donors_present"] >= MIN_DONORS_PRESENT_PER_GROUP) &
        (summary["n_Control_donors_present"] >= MIN_DONORS_PRESENT_PER_GROUP)
    )

    return summary


def plot_subcluster_proportions(sub_counts, subcluster_summary, celltype):
    """Plot only subclusters that pass the low-count filter."""
    keep_clusters = subcluster_summary.loc[
        (subcluster_summary["celltype"] == celltype) &
        (subcluster_summary["passes_low_count_filter"]),
        "subcluster"
    ].astype(str).tolist()

    plot_df = sub_counts[
        (sub_counts["celltype"] == celltype) &
        (sub_counts["subcluster"].astype(str).isin(keep_clusters))
    ].copy()

    if plot_df.empty:
        print(f"No {celltype} subclusters passed the low-count filter for plotting.")
        return

    plt.figure(figsize=(max(7, len(keep_clusters) * 1.1), 5))
    sns.boxplot(
        data=plot_df,
        x="subcluster",
        y="subcluster_proportion",
        hue="disease"
    )
    sns.stripplot(
        data=plot_df,
        x="subcluster",
        y="subcluster_proportion",
        hue="disease",
        dodge=True,
        color="black",
        alpha=0.5
    )

    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles[:2], labels[:2], title="Disease")
    plt.ylabel(f"Subcluster proportion within {celltype}")
    plt.xlabel("Within-cell-type Leiden subcluster")
    plt.title(f"{celltype}: filtered within-cell-type subcluster proportions")
    plt.tight_layout()
    plt.savefig(outdir / f"module2_{celltype}_true_subcluster_proportions_filtered.png", dpi=300)
    plt.close()


# ============================================================
# 1. Load data
# ============================================================

adata = sc.read_h5ad(h5ad_path)
adata.obs["donor"] = adata.obs["donor"].astype(str)
adata.obs["disease"] = adata.obs["disease"].astype(str).replace({
    "ct": "Control",
    "CT": "Control"
})
adata.obs["celltype"] = adata.obs["celltype"].astype(str)

adata = adata[~adata.obs["donor"].str.lower().isin(UNCLEAR_DONORS)].copy()

print("Input AnnData:")
print(adata)
print("Cell-type counts after removing unclear donors:")
print(adata.obs["celltype"].value_counts())


# ============================================================
# 2. Run true subclustering within each focal cell type
# ============================================================

all_sub_counts = []
subclustered_h5ad_paths = []

for celltype in FOCUS_CELLTYPES:
    print(f"\nProcessing focal cell type: {celltype}")

    adata_ct = adata[adata.obs["celltype"] == celltype].copy()
    print(f"{celltype} nuclei: {adata_ct.n_obs}")

    if adata_ct.n_obs < MIN_TOTAL_NUCLEI_FOR_CELLTYPE:
        print(f"Skipping {celltype}: fewer than {MIN_TOTAL_NUCLEI_FOR_CELLTYPE} nuclei.")
        continue

    adata_ct = run_subclustering(adata_ct, celltype)
    if adata_ct is None:
        continue

    # Save subclustered object for traceability.
    h5ad_out = outdir / f"module2_{celltype}_true_subclustered.h5ad"
    adata_ct.write_h5ad(h5ad_out)
    subclustered_h5ad_paths.append(str(h5ad_out))

    save_umap_plots(adata_ct, celltype)

    sub_counts_ct = count_subclusters(adata_ct, celltype)
    all_sub_counts.append(sub_counts_ct)


if not all_sub_counts:
    raise RuntimeError("No focal cell type produced a valid subclustering result.")

sub_counts = pd.concat(all_sub_counts, ignore_index=True)
sub_counts.to_csv(outdir / "module2_true_subcluster_proportions_zero_filled.csv", index=False)


# ============================================================
# 3. Low-count filter summary
# ============================================================

subcluster_summary = summarize_subclusters(sub_counts)
subcluster_summary.to_csv(outdir / "module2_true_subcluster_count_summary_with_filter.csv", index=False)

print("\nSubcluster count summary:")
print(subcluster_summary)

for celltype in FOCUS_CELLTYPES:
    plot_subcluster_proportions(sub_counts, subcluster_summary, celltype)


# ============================================================
# 4. Exploratory Mann-Whitney tests after low-count filter
# ============================================================

test_results = []

clusters_to_test = subcluster_summary[subcluster_summary["passes_low_count_filter"]].copy()

for _, row in clusters_to_test.iterrows():
    celltype = row["celltype"]
    subcluster = str(row["subcluster"])

    sub = sub_counts[
        (sub_counts["celltype"] == celltype) &
        (sub_counts["subcluster"].astype(str) == subcluster)
    ].copy()

    ad = sub[sub["disease"] == "AD"]["subcluster_proportion"]
    ctrl = sub[sub["disease"] == "Control"]["subcluster_proportion"]

    if len(ad) > 0 and len(ctrl) > 0:
        stat, pval = mannwhitneyu(ad, ctrl, alternative="two-sided")

        test_results.append({
            "celltype": celltype,
            "subcluster": subcluster,
            "mean_AD": ad.mean(),
            "mean_Control": ctrl.mean(),
            "median_AD": ad.median(),
            "median_Control": ctrl.median(),
            "difference_AD_minus_Control": ad.mean() - ctrl.mean(),
            "mannwhitney_statistic": stat,
            "p_value": pval,
            "n_AD_donors_tested": len(ad),
            "n_Control_donors_tested": len(ctrl),
            "total_nuclei": row["total_nuclei"],
            "n_donors_present": row["n_donors_present"],
            "n_AD_donors_present": row["n_AD_donors_present"],
            "n_Control_donors_present": row["n_Control_donors_present"]
        })

test_results = pd.DataFrame(test_results)

if not test_results.empty:
    test_results["q_value_BH_FDR"] = multipletests(
        test_results["p_value"],
        method="fdr_bh"
    )[1]
    test_results = test_results.sort_values("q_value_BH_FDR")

test_results.to_csv(outdir / "module2_true_subcluster_test_results_filtered_with_fdr.csv", index=False)


# ============================================================
# 5. Interpretation template
# ============================================================

interpretation = f"""
Module 2 Analysis 2 interpretation template

We performed true within-cell-type subclustering for focal AD-relevant cell types: {', '.join(FOCUS_CELLTYPES)}.
For each focal cell type, nuclei were subsetted and Leiden clustering was rerun within that cell type.
This makes the analysis distinct from the broad cell-type composition analysis in Module 2 Analysis 1.

For each donor, we calculated the proportion of each within-cell-type subcluster among that donor's nuclei of the same cell type.
Missing donor-subcluster combinations were filled with zero so that absence of a subcluster within a donor was represented explicitly.

Before statistical testing, we applied low-count filters:
- cell type must have at least {MIN_TOTAL_NUCLEI_FOR_CELLTYPE} nuclei;
- donor-cell-type combinations must have at least {MIN_NUCLEI_PER_DONOR_CELLTYPE} nuclei;
- subclusters must have at least {MIN_TOTAL_NUCLEI_PER_SUBCLUSTER} total nuclei;
- subclusters must be present in at least {MIN_DONORS_PRESENT_PER_SUBCLUSTER} donors;
- subclusters must be present in at least {MIN_DONORS_PRESENT_PER_GROUP} AD donor and {MIN_DONORS_PRESENT_PER_GROUP} Control donor.

Mann-Whitney U tests with Benjamini-Hochberg FDR correction were used as exploratory donor-level comparisons.
Because the donor sample size is limited, these results should be interpreted as suggestive evidence of within-cell-type heterogeneity rather than definitive disease-associated subtype shifts.
"""

with open(outdir / "module2_true_subcluster_interpretation_template.txt", "w") as f:
    f.write(interpretation)

print("\nModule 2 Analysis 2 true subclustering finished.")
print(f"Outputs saved to: {outdir}")
print("Subclustered h5ad files:")
for p in subclustered_h5ad_paths:
    print(p)
