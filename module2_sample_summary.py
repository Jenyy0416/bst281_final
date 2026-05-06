"""
Module 2 donor/sample summary table

Purpose:
- Summarize donor-level metadata and nuclei counts before composition modeling.
- Check whether AD and Control samples have strongly different sequencing/sampling depth.

Run from the project root:
    python module2_sample_summary.py
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# Paths and configuration
# ============================================================

project_dir = Path(__file__).resolve().parent
data_results_dir = project_dir / "data/results"
outdir = project_dir / "results/module2_sample_summary"
outdir.mkdir(parents=True, exist_ok=True)

h5ad_path = data_results_dir / "adata_embedded_fixed.h5ad"
counts_path = data_results_dir / "table_nuclei_per_donor_celltype.csv"

UNCLEAR_DONORS = ["ad-un", "ct-un"]
NON_BIOLOGICAL_CATEGORIES = ["doublet", "unID"]


# ============================================================
# 1. Load data
# ============================================================

adata = sc.read_h5ad(h5ad_path)
counts_wide = pd.read_csv(counts_path)

obs = adata.obs.copy()
obs["donor"] = obs["donor"].astype(str)
obs["disease"] = obs["disease"].astype(str).replace({
    "ct": "Control",
    "CT": "Control"
})

if "celltype" in obs.columns:
    obs["celltype"] = obs["celltype"].astype(str)

counts_wide["donor"] = counts_wide["donor"].astype(str)

# Remove unclear donors consistently
obs = obs[~obs["donor"].str.lower().isin(UNCLEAR_DONORS)].copy()
counts_wide = counts_wide[~counts_wide["donor"].str.lower().isin(UNCLEAR_DONORS)].copy()


# ============================================================
# 2. Donor metadata
# ============================================================

metadata_cols = ["donor", "disease"]
for optional_col in ["sex", "age", "batch", "sample", "region"]:
    if optional_col in obs.columns:
        metadata_cols.append(optional_col)

# If optional metadata are not donor-unique, keep only donor and disease.
donor_info = obs[metadata_cols].drop_duplicates().reset_index(drop=True)
if donor_info["donor"].duplicated().any():
    print("Warning: donor metadata are not unique. Keeping only donor and disease.")
    donor_info = obs[["donor", "disease"]].drop_duplicates().reset_index(drop=True)


# ============================================================
# 3. Count summary from wide count table
# ============================================================

count_cols_all = [c for c in counts_wide.columns if c != "donor"]
count_cols_bio = [c for c in count_cols_all if c not in NON_BIOLOGICAL_CATEGORIES]

# Ensure count columns are numeric
for c in count_cols_all:
    counts_wide[c] = pd.to_numeric(counts_wide[c], errors="coerce").fillna(0)

sample_summary = counts_wide[["donor"] + count_cols_all].copy()
sample_summary["total_nuclei_all_categories"] = sample_summary[count_cols_all].sum(axis=1)
sample_summary["total_nuclei_biological"] = sample_summary[count_cols_bio].sum(axis=1)

if "doublet" in sample_summary.columns:
    sample_summary["doublet_fraction"] = (
        sample_summary["doublet"] / sample_summary["total_nuclei_all_categories"].replace(0, np.nan)
    )

if "unID" in sample_summary.columns:
    sample_summary["unID_fraction"] = (
        sample_summary["unID"] / sample_summary["total_nuclei_all_categories"].replace(0, np.nan)
    )

sample_summary = donor_info.merge(sample_summary, on="donor", how="right")

# Reorder columns for readability
front_cols = [c for c in [
    "donor", "disease", "sex", "age", "batch", "sample", "region",
    "total_nuclei_all_categories", "total_nuclei_biological",
    "doublet_fraction", "unID_fraction"
] if c in sample_summary.columns]
other_cols = [c for c in sample_summary.columns if c not in front_cols]
sample_summary = sample_summary[front_cols + other_cols]

sample_summary.to_csv(outdir / "module2_donor_sample_summary.csv", index=False)


# ============================================================
# 4. Disease-level sample-depth summary
# ============================================================

summary_vars = ["total_nuclei_all_categories", "total_nuclei_biological"]
for optional_var in ["doublet_fraction", "unID_fraction"]:
    if optional_var in sample_summary.columns:
        summary_vars.append(optional_var)

agg_dict = {
    "donor": "nunique"
}
for var in summary_vars:
    agg_dict[var] = ["mean", "median", "std", "min", "max"]

disease_summary = sample_summary.groupby("disease").agg(agg_dict)
disease_summary.columns = [
    "_".join([str(x) for x in col if x]) for col in disease_summary.columns
]
disease_summary = disease_summary.rename(columns={"donor_nunique": "n_donors"}).reset_index()
disease_summary.to_csv(outdir / "module2_sample_depth_summary_by_disease.csv", index=False)


# ============================================================
# 5. Cell-type count summary by disease
# ============================================================

counts_long = counts_wide.melt(
    id_vars="donor",
    value_vars=count_cols_bio,
    var_name="celltype",
    value_name="n_nuclei"
)
counts_long = counts_long.merge(donor_info[["donor", "disease"]], on="donor", how="left")

celltype_count_summary = (
    counts_long
    .groupby(["disease", "celltype"])
    .agg(
        total_nuclei=("n_nuclei", "sum"),
        mean_nuclei_per_donor=("n_nuclei", "mean"),
        median_nuclei_per_donor=("n_nuclei", "median"),
        n_donors=("donor", "nunique"),
        n_donors_present=("n_nuclei", lambda x: int((x > 0).sum()))
    )
    .reset_index()
)
celltype_count_summary.to_csv(outdir / "module2_celltype_count_summary_by_disease.csv", index=False)


# ============================================================
# 6. Plots
# ============================================================

plt.figure(figsize=(7, 5))
sns.boxplot(
    data=sample_summary,
    x="disease",
    y="total_nuclei_biological"
)
sns.stripplot(
    data=sample_summary,
    x="disease",
    y="total_nuclei_biological",
    color="black",
    alpha=0.6
)
plt.ylabel("Total biological nuclei per donor")
plt.xlabel("Disease group")
plt.title("Sample depth by disease group")
plt.tight_layout()
plt.savefig(outdir / "module2_total_biological_nuclei_by_disease.png", dpi=300)
plt.close()

# Optional heatmap of biological cell-type counts by donor
heatmap_df = sample_summary.set_index("donor")[count_cols_bio]
plt.figure(figsize=(max(8, len(count_cols_bio) * 0.8), max(4, heatmap_df.shape[0] * 0.35)))
sns.heatmap(heatmap_df, cmap="viridis")
plt.xlabel("Cell type")
plt.ylabel("Donor")
plt.title("Biological nuclei counts by donor and cell type")
plt.tight_layout()
plt.savefig(outdir / "module2_donor_celltype_count_heatmap.png", dpi=300)
plt.close()


# ============================================================
# 7. Interpretation template
# ============================================================

interpretation = """
Module 2 donor/sample summary interpretation template

Before compositional testing, we summarized donor-level nuclei counts and disease labels.
This table is used to check whether AD and Control donors have comparable sample depth.
The main composition analyses exclude doublets and unidentified nuclei, but the summary table also reports these categories for transparency.

If one disease group has much larger total nuclei counts than the other, downstream cell-type proportion results should be interpreted cautiously because sampling depth can affect the observed number of rare cell types.
"""

with open(outdir / "module2_sample_summary_interpretation_template.txt", "w") as f:
    f.write(interpretation)

print("Module 2 sample summary finished.")
print(f"Outputs saved to: {outdir}")
