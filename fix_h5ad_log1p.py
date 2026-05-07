import shutil
import h5py
from pathlib import Path

input_h5ad = Path("data/adata_embedded.h5ad")
fixed_h5ad = Path("data/adata_embedded_fixed.h5ad")

# Copy original file so we do not modify it directly
shutil.copy2(input_h5ad, fixed_h5ad)

# Remove problematic /uns/log1p/base field
with h5py.File(fixed_h5ad, "r+") as f:
    if "uns/log1p/base" in f:
        del f["uns/log1p/base"]
        print("Deleted uns/log1p/base")
    else:
        print("uns/log1p/base not found; no change needed")

print(f"Saved fixed file as: {fixed_h5ad}")