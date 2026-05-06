import os
import shutil
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", ".cache/matplotlib")
os.environ.setdefault("NUMBA_CACHE_DIR", ".cache/numba")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)
os.makedirs(os.environ["NUMBA_CACHE_DIR"], exist_ok=True)

import h5py
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests


# ============================================================
# Project paths
# ============================================================

project_dir = Path(__file__).resolve().parent

data_results_dir = project_dir / "data/results"
outdir = project_dir / "results/module2_analysis1"
outdir.mkdir(parents=True, exist_ok=True)

input_h5ad = data_results_dir / "adata_embedded.h5ad"
fixed_h5ad = data_results_dir / "adata_embedded_fixed.h5ad"
counts_path = data_results_dir / "table_nuclei_per_donor_celltype.csv"


# ============================================================
# Fix h5ad log1p issue if needed
# ============================================================

if not fixed_h5ad.exists():
    shutil.copy2(input_h5ad, fixed_h5ad)

    with h5py.File(fixed_h5ad, "r+") as f:
        if "uns/log1p/base" in f:
            del f["uns/log1p/base"]


# ============================================================
# 1. Load data
# ============================================================

adata = sc.read_h5ad(fixed_h5ad)
counts_wide = pd.read_csv(counts_path)

# Standardize labels
adata.obs["donor"] = adata.obs["donor"].astype(str)
adata.obs["disease"] = adata.obs["disease"].astype(str).replace({
    "ct": "Control",
    "CT": "Control"
})

counts_wide["donor"] = counts_wide["donor"].astype(str)

print("AnnData object:")
print(adata)

print("\nAnnData obs columns:")
print(adata.obs.columns)

print("\nOriginal count table:")
print(counts_wide.head())
print(counts_wide.columns)


# ============================================================
# 2. Remove unclear donors
# ============================================================

unclear_donors = ["ad-un", "ct-un"]

adata = adata[
    ~adata.obs["donor"].str.lower().isin(unclear_donors)
].copy()

counts_wide = counts_wide[
    ~counts_wide["donor"].str.lower().isin(unclear_donors)
].copy()

print("\nDonors after removing unclear IDs:")
print(sorted(counts_wide["donor"].unique()))


# ============================================================
# 3. Convert wide count table to long format
# ============================================================

counts_long = counts_wide.melt(
    id_vars="donor",
    var_name="celltype",
    value_name="n_nuclei"
)

# Remove doublets and unidentified cells
counts_long = counts_long[
    ~counts_long["celltype"].isin(["doublet", "unID"])
].copy()


# ============================================================
# 4. Add donor-level metadata
# ============================================================

donor_info = (
    adata.obs[["donor", "disease", "sex"]]
    .drop_duplicates()
    .reset_index(drop=True)
)

# If one donor has duplicated sex labels, keep donor and disease only
# This avoids accidental duplicate rows in donor-level analysis.
if donor_info["donor"].duplicated().any():
    print("\nWarning: duplicated donor metadata detected. Keeping donor and disease only.")
    donor_info = (
        adata.obs[["donor", "disease"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )

counts_long = counts_long.merge(donor_info, on="donor", how="left")

print("\nLong-format count table:")
print(counts_long.head())

print("\nDisease group counts in long table:")
print(counts_long["disease"].value_counts())


# ============================================================
# 5. Calculate donor-level cell-type proportions
# ============================================================

counts_long["total_nuclei_per_donor"] = (
    counts_long
    .groupby("donor")["n_nuclei"]
    .transform("sum")
)

counts_long["proportion"] = (
    counts_long["n_nuclei"] / counts_long["total_nuclei_per_donor"]
)

counts_long.to_csv(
    outdir / "module2_donor_celltype_proportions.csv",
    index=False
)

print("\nDonor-level proportions:")
print(counts_long.head())


# ============================================================
# 6. Save summary table by disease and cell type
# ============================================================

summary_table = (
    counts_long
    .groupby(["disease", "celltype"])
    .agg(
        mean_proportion=("proportion", "mean"),
        median_proportion=("proportion", "median"),
        sd_proportion=("proportion", "std"),
        mean_n_nuclei=("n_nuclei", "mean"),
        total_n_nuclei=("n_nuclei", "sum"),
        n_donors=("donor", "nunique")
    )
    .reset_index()
)

summary_table.to_csv(
    outdir / "module2_celltype_summary_by_disease.csv",
    index=False
)

print("\nSummary table by disease and cell type:")
print(summary_table)


# ============================================================
# 7. Plot 1: stacked bar plot by donor
# ============================================================

prop_wide = counts_long.pivot_table(
    index="donor",
    columns="celltype",
    values="proportion",
    fill_value=0
)

donor_order = (
    counts_long[["donor", "disease"]]
    .drop_duplicates()
    .sort_values(["disease", "donor"])["donor"]
    .tolist()
)

donor_order = [d for d in donor_order if d in prop_wide.index]
prop_wide = prop_wide.loc[donor_order]

ax = prop_wide.plot(
    kind="bar",
    stacked=True,
    figsize=(12, 6)
)

plt.ylabel("Cell-type proportion")
plt.xlabel("Donor")
plt.title("Cell-type composition per donor")
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.savefig(
    outdir / "module2_celltype_composition_per_donor.png",
    dpi=300
)
plt.close()


# ============================================================
# 8. Plot 2: mean composition by disease group
# ============================================================

group_summary = (
    counts_long
    .groupby(["disease", "celltype"])["proportion"]
    .mean()
    .reset_index()
)

group_wide = group_summary.pivot_table(
    index="disease",
    columns="celltype",
    values="proportion",
    fill_value=0
)

ax = group_wide.plot(
    kind="bar",
    stacked=True,
    figsize=(8, 5)
)

plt.ylabel("Mean donor-level cell-type proportion")
plt.xlabel("Disease group")
plt.title("Mean cell-type composition by disease group")
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.savefig(
    outdir / "module2_mean_composition_by_disease.png",
    dpi=300
)
plt.close()


# ============================================================
# 9. Plot 3: focal AD-relevant cell types
# ============================================================

focus_celltypes = ["neuron", "astro", "mg"]

focus_counts = counts_long[
    counts_long["celltype"].isin(focus_celltypes)
].copy()

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=focus_counts,
    x="celltype",
    y="proportion",
    hue="disease"
)

sns.stripplot(
    data=focus_counts,
    x="celltype",
    y="proportion",
    hue="disease",
    dodge=True,
    color="black",
    alpha=0.5
)

# Remove duplicated legend entries caused by boxplot + stripplot
handles, labels = plt.gca().get_legend_handles_labels()
plt.legend(handles[:2], labels[:2], title="Disease")

plt.ylabel("Donor-level proportion")
plt.xlabel("Cell type")
plt.title("Donor-level proportions of AD-relevant cell types")
plt.tight_layout()
plt.savefig(
    outdir / "module2_focal_celltype_proportions.png",
    dpi=300
)
plt.close()


# ============================================================
# 9b. Plot 4: top composition-shift cell types
# ============================================================

top_celltypes = ["OPC", "astro", "oligo", "neuron"]

top_counts = counts_long[
    counts_long["celltype"].isin(top_celltypes)
].copy()

plt.figure(figsize=(9, 5))

sns.boxplot(
    data=top_counts,
    x="celltype",
    y="proportion",
    hue="disease"
)

sns.stripplot(
    data=top_counts,
    x="celltype",
    y="proportion",
    hue="disease",
    dodge=True,
    color="black",
    alpha=0.5
)

# Remove duplicated legend entries caused by boxplot + stripplot
handles, labels = plt.gca().get_legend_handles_labels()
plt.legend(handles[:2], labels[:2], title="Disease")

plt.ylabel("Donor-level proportion")
plt.xlabel("Cell type")
plt.title("Donor-level proportions of top composition-shift cell types")
plt.tight_layout()
plt.savefig(
    outdir / "module2_top_shift_celltype_proportions.png",
    dpi=300
)
plt.close()


# ============================================================
# 10. Exploratory donor-level Mann-Whitney U tests
# ============================================================

test_results = []

for ct in sorted(counts_long["celltype"].unique()):
    sub = counts_long[counts_long["celltype"] == ct].copy()

    ad = sub[sub["disease"] == "AD"]["proportion"]
    ctrl = sub[sub["disease"] == "Control"]["proportion"]

    if len(ad) > 0 and len(ctrl) > 0:
        stat, pval = mannwhitneyu(
            ad,
            ctrl,
            alternative="two-sided"
        )

        mean_ad = ad.mean()
        mean_ctrl = ctrl.mean()

        test_results.append({
            "celltype": ct,
            "mean_AD": mean_ad,
            "mean_Control": mean_ctrl,
            "median_AD": ad.median(),
            "median_Control": ctrl.median(),
            "difference_AD_minus_Control": mean_ad - mean_ctrl,
            "ratio_AD_over_Control": mean_ad / mean_ctrl if mean_ctrl != 0 else np.nan,
            "log2_ratio_AD_over_Control": np.log2(
                (mean_ad + 1e-6) / (mean_ctrl + 1e-6)
            ),
            "mannwhitney_statistic": stat,
            "p_value": pval,
            "n_AD_donors": len(ad),
            "n_Control_donors": len(ctrl)
        })

test_results = pd.DataFrame(test_results)

# FDR correction
if not test_results.empty:
    test_results["q_value_BH_FDR"] = multipletests(
        test_results["p_value"],
        method="fdr_bh"
    )[1]

    test_results = test_results.sort_values("q_value_BH_FDR")

test_results.to_csv(
    outdir / "module2_composition_test_results_with_fdr.csv",
    index=False
)

print("\nExploratory donor-level Mann-Whitney results with FDR:")
print(test_results)


# ============================================================
# 11. Save a short text interpretation template
# ============================================================

interpretation = """
Module 2 Analysis 1 interpretation template

We summarized broad cell-type composition at the donor level using QC-passed annotated nuclei from Module 1.
For each donor, we calculated the proportion of each annotated biological cell type among all retained nuclei.
Doublets and unidentified nuclei were excluded from the main biological composition analysis.
Donors with unclear identifiers were excluded from donor-level statistical analyses.

Exploratory Mann-Whitney U tests were used to compare donor-level cell-type proportions between AD and control donors.
Because multiple cell types were tested, Benjamini-Hochberg FDR correction was applied.
These tests are interpreted as descriptive and exploratory because the number of donors is limited.

The scCODA compositional model should be treated as the main compositional analysis method.
In our scCODA analysis, diagnosis was not selected as a credible effect for any broad annotated cell type.
Therefore, broad cell-type composition differences should be interpreted as suggestive rather than definitive.
"""

with open(outdir / "module2_analysis1_interpretation_template.txt", "w") as f:
    f.write(interpretation)


print("\nModule 2 Analysis 1 finished.")
print(f"Outputs saved to: {outdir}")
