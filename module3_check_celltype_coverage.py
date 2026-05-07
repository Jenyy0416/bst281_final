import anndata as ad
import pandas as pd

# Load fixed AnnData
adata = ad.read_h5ad("data/adata_embedded_fixed.h5ad")

# Metadata
obs = adata.obs.copy()
obs["donor"] = obs["donor"].astype(str)
obs["disease"] = obs["disease"].astype(str).replace({"ct": "Control", "CT": "Control"})
obs["celltype"] = obs["celltype"].astype(str)
obs["sex"] = obs["sex"].astype(str)

# Remove unclear donors
unclear_donors = ["AD-un", "Ct-un", "ad-un", "ct-un"]
obs = obs[~obs["donor"].isin(unclear_donors)].copy()

# Focal cell types
focal_celltypes = ["astro", "neuron", "mg"]
obs_focal = obs[obs["celltype"].isin(focal_celltypes)].copy()

# Correct donor-level count table: only observed donor/disease/sex/celltype combinations
coverage = (
    obs_focal
    .groupby(["celltype", "donor", "disease", "sex"], observed=True)
    .size()
    .reset_index(name="n_nuclei")
)

coverage.to_csv("module3_focal_celltype_donor_coverage.csv", index=False)

# Summary by cell type and disease
summary = (
    coverage
    .groupby(["celltype", "disease"], observed=True)
    .agg(
        total_nuclei=("n_nuclei", "sum"),
        n_donors_present=("donor", "nunique"),
        mean_nuclei_per_donor=("n_nuclei", "mean"),
        median_nuclei_per_donor=("n_nuclei", "median"),
        min_nuclei_per_donor=("n_nuclei", "min"),
        max_nuclei_per_donor=("n_nuclei", "max")
    )
    .reset_index()
)

summary.to_csv("module3_focal_celltype_coverage_summary.csv", index=False)

# Wide table
wide = coverage.pivot_table(
    index=["donor", "disease", "sex"],
    columns="celltype",
    values="n_nuclei",
    fill_value=0,
    observed=True
).reset_index()

wide.to_csv("module3_focal_celltype_coverage_wide.csv", index=False)

print("\nCorrected detailed donor-level coverage:")
print(coverage.sort_values(["celltype", "disease", "donor"]).to_string(index=False))

print("\nCorrected coverage summary:")
print(summary.to_string(index=False))

print("\nCorrected wide donor coverage table:")
print(wide.to_string(index=False))