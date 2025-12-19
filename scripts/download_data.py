"""
Script to download TCGA-LUAD dataset from UCSC Xena.
Downloads both gene expression data and clinical data (if available).
"""

import sys
from pathlib import Path

# Add backend to path
sys.path.append(str(Path(__file__).parent.parent))

import requests
import gzip
import shutil
from tqdm import tqdm
from backend.config import (
    RAW_DATA_DIR,
    TCGA_LUAD_URL,
    TCGA_LUAD_CLINICAL_URL
)

def download_file(url: str, output_path: Path, decompress: bool = True) -> bool:
    """
    Download file from URL with progress bar.
    
    Args:
        url: URL to download from
        output_path: Path to save file
        decompress: Whether to decompress .gz files
        
    Returns:
        True if successful, False otherwise
    """
    try:
        print(f"\nDownloading from: {url}")
        print(f"Saving to: {output_path}")
        
        # Create output directory
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Download with progress bar
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        # Determine temporary filename
        if decompress and url.endswith('.gz'):
            temp_path = output_path.with_suffix(output_path.suffix + '.gz')
        else:
            temp_path = output_path
        
        # Download file
        with open(temp_path, 'wb') as f, tqdm(
            desc="Downloading",
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
        
        print(f"✓ Download complete: {temp_path}")
        
        # Decompress if needed
        if decompress and url.endswith('.gz'):
            print(f"Decompressing {temp_path.name}...")
            with gzip.open(temp_path, 'rb') as f_in:
                with open(output_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            
            # Remove compressed file
            temp_path.unlink()
            print(f"✓ Decompressed to: {output_path}")
        
        # Verify file exists and has size > 0
        if output_path.exists() and output_path.stat().st_size > 0:
            file_size_mb = output_path.stat().st_size / (1024 * 1024)
            print(f"✓ File verified: {file_size_mb:.2f} MB")
            return True
        else:
            print(f"✗ Error: Downloaded file is empty or missing")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"✗ Download error: {e}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False

def main():
    """Main download function."""
    print("=" * 70)
    print("TCGA-LUAD Data Download Script")
    print("=" * 70)
    
    # Check if data directory exists
    if not RAW_DATA_DIR.exists():
        RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
        print(f"Created data directory: {RAW_DATA_DIR}")
    
    # Download gene expression data
    gene_expr_file = RAW_DATA_DIR / "TCGA-LUAD.star_tpm.tsv"
    
    if gene_expr_file.exists():
        print(f"\n✓ Gene expression data already exists: {gene_expr_file}")
        overwrite = input("Do you want to re-download? (y/n): ").lower()
        if overwrite != 'y':
            print("Skipping gene expression data download.")
        else:
            success = download_file(TCGA_LUAD_URL, gene_expr_file, decompress=True)
            if not success:
                print("Failed to download gene expression data.")
                return
    else:
        success = download_file(TCGA_LUAD_URL, gene_expr_file, decompress=True)
        if not success:
            print("Failed to download gene expression data.")
            return
    
    # Download clinical data (optional)
    print("\n" + "-" * 70)
    print("Clinical Data Download (Optional)")
    print("-" * 70)
    
    clinical_file = RAW_DATA_DIR / "TCGA-LUAD.clinical.tsv"
    
    download_clinical = input("\nDo you want to download clinical data? (y/n): ").lower()
    
    if download_clinical == 'y':
        if clinical_file.exists():
            print(f"\n✓ Clinical data already exists: {clinical_file}")
            overwrite = input("Do you want to re-download? (y/n): ").lower()
            if overwrite == 'y':
                success = download_file(TCGA_LUAD_CLINICAL_URL, clinical_file, decompress=True)
                if not success:
                    print("Failed to download clinical data (non-critical).")
        else:
            success = download_file(TCGA_LUAD_CLINICAL_URL, clinical_file, decompress=True)
            if not success:
                print("Failed to download clinical data (non-critical).")
    else:
        print("Skipping clinical data download.")
    
    # Summary
    print("\n" + "=" * 70)
    print("Download Summary")
    print("=" * 70)
    
    files_downloaded = []
    if gene_expr_file.exists():
        files_downloaded.append(("Gene Expression", gene_expr_file))
    if clinical_file.exists():
        files_downloaded.append(("Clinical Data", clinical_file))
    
    if files_downloaded:
        print("\n✓ Downloaded files:")
        for name, path in files_downloaded:
            size_mb = path.stat().st_size / (1024 * 1024)
            print(f"  - {name}: {path.name} ({size_mb:.2f} MB)")
    else:
        print("\n✗ No files were downloaded.")
    
    print("\n" + "=" * 70)
    print("Next steps:")
    print("1. Run data exploration notebook: notebooks/01_data_exploration.ipynb")
    print("2. Or proceed to Phase 2: Data Preprocessing")
    print("=" * 70)

if __name__ == "__main__":
    main()