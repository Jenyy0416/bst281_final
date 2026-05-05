import os
import zipfile
import requests
from pathlib import Path

# Confidence: 1.0 - Standard automation logic
def setup_data():
    """
    Automates data downloading and extraction for the AD project.
    """
    # 1. Define paths and URLs
    base_dir = Path(__file__).parent
    data_raw_dir = base_dir / "data" / "raw"
    
    # Files from Zenodo record 17302976
    files_to_download = {
        "adsn_metadata.txt": "https://zenodo.org/records/17302976/files/adsn_metadata.txt?download=1",
        "adsn_mtx.zip": "https://zenodo.org/records/17302976/files/adsn_matrix.mtx.gz?download=1"
    }

    # 2. Create directories
    data_raw_dir.mkdir(parents=True, exist_ok=True)
    print(f"Directory ready: {data_raw_dir}")

    # 3. Download files
    for file_name, url in files_to_download.items():
        file_path = data_raw_dir / file_name
        
        if file_path.exists():
            print(f"Skipping {file_name}, already exists.")
            continue
            
        print(f"Downloading {file_name}...")
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            with open(file_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"Successfully downloaded {file_name}")
        except Exception as e:
            print(f"Error downloading {file_name}: {e}")
            return

    # 4. Extract mtx zip
    zip_path = data_raw_dir / "adsn_mtx.zip"
    extract_path = data_raw_dir / "adsn_mtx"
    
    if zip_path.exists() and not extract_path.exists():
        print("Extracting adsn_mtx.zip...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_file.extractall(extract_path)
        print(f"Extracted to {extract_path}")

    print("\nData pipeline setup complete!")
    print(f"Metadata: {data_raw_dir / 'adsn_metadata.txt'}")
    print(f"Matrix files: {extract_path}")

if __name__ == "__main__":
    setup_data()
