import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from pathlib import Path

# =========================
# Settings
# =========================

input_h5ad = "data/adata_embedded_fixed.h5ad"
outdir = Path("module3_astro")
outdir.mkdir(parents=True, exist_ok=True)

celltype_of_interest = "astro"
unclear_donors = ["AD-un", "Ct-un", "ad-un", "ct-un"]

# =========================
# Load AnnData
# =========================

adata = ad.read_h5ad(input_h5ad)

adata.obs["donor"] = adata.obs["donor"].astype(str)
adata.obs["disease"] = adata.obs["disease"].astype(str).replace({
    "ct": "Control",
    "CT": "Control"
})
adata.obs["celltype"] = adata.obs["celltype"].astype(str)
adata.obs["sex"] = adata.obs["sex"].astype(str)

# Remove unclear donors
adata = adata[~adata.obs["donor"].isin(unclear_donors)].copy()

# Subset to astrocytes
adata_ct = adata[adata.obs["celltype"] == celltype_of_interest].copy()

print(adata_ct)
print(adata_ct.obs.groupby(["donor", "disease", "sex"], observed=True).size())

# =========================
# Use raw counts
# =========================

if "counts" not in adata_ct.layers.keys():
    raise ValueError("No adata.layers['counts'] found.")

X = adata_ct.layers["counts"]

if not sp.issparse(X):
    X = sp.csr_matrix(X)
else:
    X = X.tocsr()

genes = pd.Index(adata_ct.var_names)
donors = sorted(adata_ct.obs["donor"].unique())

# =========================
# Aggregate counts by donor
# =========================

pseudobulk_list = []
metadata_list = []

for donor in donors:
    mask = (adata_ct.obs["donor"] == donor).values
    counts = np.asarray(X[mask, :].sum(axis=0)).ravel()

    pseudobulk_list.append(counts)

    donor_meta = (
        adata_ct.obs.loc[mask, ["donor", "disease", "sex"]]
        .drop_duplicates()
    )

    if donor_meta.shape[0] != 1:
        raise ValueError(f"Non-unique metadata for donor {donor}:\n{donor_meta}")

    metadata_list.append({
        "donor": donor,
        "disease": donor_meta["disease"].iloc[0],
        "sex": donor_meta["sex"].iloc[0],
        "n_nuclei": int(mask.sum()),
        "library_size": int(counts.sum())
    })

# DESeq2 wants genes as rows and samples as columns
pseudobulk_counts = pd.DataFrame(
    pseudobulk_list,
    index=donors,
    columns=genes
).T

sample_metadata = pd.DataFrame(metadata_list).set_index("donor")

# Save
pseudobulk_counts.to_csv(outdir / "astro_pseudobulk_counts_genes_by_donor.csv")
sample_metadata.to_csv(outdir / "astro_sample_metadata.csv")

print("Saved:")
print(outdir / "astro_pseudobulk_counts_genes_by_donor.csv")
print(outdir / "astro_sample_metadata.csv")