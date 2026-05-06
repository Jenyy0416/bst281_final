"""
Module 2 scCODA with reference-cell-type sensitivity analysis

Purpose:
- Use scCODA as the main broad cell-type compositional analysis.
- Repeat the model with several reference cell types to check whether the conclusion
  depends on the reference choice.

Run from the project root:
    python module2_scCODA_sensitivity.py
"""

import os
from pathlib import Path
from contextlib import redirect_stdout

os.environ.setdefault("MPLCONFIGDIR", ".cache/matplotlib")
os.environ.setdefault("NUMBA_CACHE_DIR", ".cache/numba")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)
os.makedirs(os.environ["NUMBA_CACHE_DIR"], exist_ok=True)

import pandas as pd
import scanpy as sc

from sccoda.util import comp_ana as mod
from sccoda.util import cell_composition_data as dat


# ============================================================
# Paths and configuration
# ============================================================

project_dir = Path(__file__).resolve().parent
data_results_dir = project_dir / "data/results"
outdir = project_dir / "results/module2_sccoda"
outdir.mkdir(parents=True, exist_ok=True)

h5ad_path = data_results_dir / "adata_embedded_fixed.h5ad"
counts_path = data_results_dir / "table_nuclei_per_donor_celltype.csv"

UNCLEAR_DONORS = ["ad-un", "ct-un"]
NON_BIOLOGICAL_CATEGORIES = ["doublet", "unID"]

# Main reference plus sensitivity references.
# The script automatically skips references that are not present in the count table.
REFERENCE_CELL_TYPES = ["mg", "oligo", "astro", "OPC"]


# ============================================================
# 1. Load and prepare data
# ============================================================

adata = sc.read_h5ad(h5ad_path)
counts_wide = pd.read_csv(counts_path)

obs = adata.obs.copy()
obs["donor"] = obs["donor"].astype(str)
obs["disease"] = obs["disease"].astype(str).replace({
    "ct": "Control",
    "CT": "Control"
})

counts_wide["donor"] = counts_wide["donor"].astype(str)

counts_wide = counts_wide[~counts_wide["donor"].str.lower().isin(UNCLEAR_DONORS)].copy()
obs = obs[~obs["donor"].str.lower().isin(UNCLEAR_DONORS)].copy()

donor_info = (
    obs[["donor", "disease"]]
    .drop_duplicates()
    .reset_index(drop=True)
)

if donor_info["donor"].duplicated().any():
    raise ValueError("Donor metadata are not unique after keeping donor and disease.")

counts_sccoda = counts_wide.copy()
drop_cols = [c for c in NON_BIOLOGICAL_CATEGORIES if c in counts_sccoda.columns]
counts_sccoda = counts_sccoda.drop(columns=drop_cols)

counts_sccoda = counts_sccoda.merge(donor_info, on="donor", how="left")
if counts_sccoda["disease"].isna().any():
    print("Warning: Some donors have missing disease labels:")
    print(counts_sccoda[counts_sccoda["disease"].isna()])

counts_sccoda["disease_AD"] = (counts_sccoda["disease"] == "AD").astype(int)
counts_sccoda = counts_sccoda.drop(columns=["disease"])
counts_sccoda = counts_sccoda.set_index("donor")

# Ensure cell count columns are numeric integers where possible
for c in counts_sccoda.columns:
    if c != "disease_AD":
        counts_sccoda[c] = pd.to_numeric(counts_sccoda[c], errors="coerce").fillna(0).astype(int)

counts_sccoda.to_csv(outdir / "module2_sccoda_input_table.csv")
print("Saved scCODA input table.")
print(counts_sccoda)

celltype_cols = [c for c in counts_sccoda.columns if c != "disease_AD"]
available_refs = [r for r in REFERENCE_CELL_TYPES if r in celltype_cols]

if not available_refs:
    raise ValueError("None of the requested reference cell types were found in the scCODA input table.")


# ============================================================
# 2. Run scCODA for each reference cell type
# ============================================================

run_records = []

data_all = dat.from_pandas(
    counts_sccoda,
    covariate_columns=["disease_AD"]
)

for ref in available_refs:
    print(f"\nRunning scCODA with reference cell type: {ref}")

    ref_outdir = outdir / f"reference_{ref}"
    ref_outdir.mkdir(parents=True, exist_ok=True)

    record = {
        "reference_cell_type": ref,
        "status": "not_run",
        "summary_file": str(ref_outdir / f"module2_sccoda_summary_ref_{ref}.txt")
    }

    try:
        model = mod.CompositionalAnalysis(
            data_all,
            formula="disease_AD",
            reference_cell_type=ref
        )

        sim_results = model.sample_hmc()

        # Save printed summary
        summary_path = ref_outdir / f"module2_sccoda_summary_ref_{ref}.txt"
        with open(summary_path, "w") as f:
            with redirect_stdout(f):
                sim_results.summary()

        # Try to save available result tables. scCODA object attributes can vary by version.
        possible_attrs = [
            "effect_df",
            "intercept_df",
            "covariate_df",
            "cell_type_df"
        ]
        for attr in possible_attrs:
            if hasattr(sim_results, attr):
                obj = getattr(sim_results, attr)
                if isinstance(obj, pd.DataFrame):
                    obj.to_csv(ref_outdir / f"module2_sccoda_{attr}_ref_{ref}.csv")

        record["status"] = "success"
        print(f"Finished scCODA reference {ref}")

    except Exception as e:
        record["status"] = "failed"
        record["error"] = str(e)
        print(f"scCODA failed for reference {ref}: {e}")

    run_records.append(record)

run_log = pd.DataFrame(run_records)
run_log.to_csv(outdir / "module2_sccoda_reference_sensitivity_run_log.csv", index=False)


# ============================================================
# 3. Interpretation template
# ============================================================

interpretation = """
Module 2 scCODA reference sensitivity interpretation template

We used scCODA as the main broad cell-type compositional analysis because cell-type proportions are compositional.
The main model used diagnosis as the covariate and modeled AD status as disease_AD, with Control coded as 0 and AD coded as 1.
To check whether the conclusion depended on the selected reference cell type, we repeated the model using several biologically annotated reference cell types.

If the same conclusion is obtained across references, we can state that the broad compositional finding is robust to the reference-cell-type choice.
If results differ by reference, the report should emphasize the main reference model and describe the sensitivity analysis as a limitation.
"""

with open(outdir / "module2_sccoda_reference_sensitivity_interpretation_template.txt", "w") as f:
    f.write(interpretation)

print("\nscCODA reference sensitivity analysis finished.")
print(f"Outputs saved to: {outdir}")
