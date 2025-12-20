"""
Data loader module for TCGA-LUAD dataset.
Handles loading and initial validation of raw data.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import pandas as pd
import numpy as np
from typing import Tuple, Optional, List
from backend.config import RAW_DATA_DIR, TUMOR_SAMPLE_SUFFIX, NORMAL_SAMPLE_SUFFIX
from backend.utils import setup_logging
import logging


class TCGALUADLoader:
    """
    Loader for TCGA-LUAD gene expression data.
    """
    
    def __init__(
        self,
        data_file: Optional[Path] = None,
        clinical_file: Optional[Path] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize TCGA-LUAD data loader.
        
        Args:
            data_file: Path to gene expression TSV file
            clinical_file: Path to clinical data TSV file (optional)
            logger: Logger instance
        """
        self.data_file = data_file or (RAW_DATA_DIR / "TCGA-LUAD.star_tpm.tsv")
        self.clinical_file = clinical_file
        self.logger = logger or logging.getLogger(__name__)
        
        self.data = None
        self.clinical_data = None
        self.sample_types = None
        
    def load_gene_expression(self) -> pd.DataFrame:
        """
        Load gene expression data from TSV file.
        
        Returns:
            DataFrame with genes as rows, samples as columns
        """
        self.logger.info(f"Loading gene expression data from: {self.data_file}")
        
        if not self.data_file.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_file}")
        
        # Load data
        self.data = pd.read_csv(self.data_file, sep='\t', index_col=0)
        
        self.logger.info(f"Loaded data shape: {self.data.shape} (genes × samples)")
        self.logger.info(f"Number of genes: {self.data.shape[0]:,}")
        self.logger.info(f"Number of samples: {self.data.shape[1]:,}")
        
        return self.data
    
    def load_clinical_data(self) -> Optional[pd.DataFrame]:
        """
        Load clinical data if available.
        
        Returns:
            DataFrame with clinical information, or None if not available
        """
        if self.clinical_file is None or not self.clinical_file.exists():
            self.logger.warning("Clinical data file not found or not specified.")
            return None
        
        self.logger.info(f"Loading clinical data from: {self.clinical_file}")
        self.clinical_data = pd.read_csv(self.clinical_file, sep='\t')
        
        self.logger.info(f"Loaded clinical data shape: {self.clinical_data.shape}")
        
        return self.clinical_data
    
    def identify_sample_types(self) -> pd.Series:
        """
        Identify sample types from TCGA barcodes.
        
        TCGA barcode format: TCGA-XX-XXXX-XXY
        - 01A = Primary Solid Tumor
        - 11A = Solid Tissue Normal
        
        Returns:
            Series with sample types for each column
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_gene_expression() first.")
        
        self.logger.info("Identifying sample types from TCGA barcodes...")
        
        sample_types = []
        for col in self.data.columns:
            parts = col.split('-')
            if len(parts) >= 4:
                sample_type_code = parts[3][:3]  # First 3 chars (e.g., "01A")
                
                if sample_type_code.startswith('01'):
                    sample_types.append('Tumor')
                elif sample_type_code.startswith('11'):
                    sample_types.append('Normal')
                else:
                    sample_types.append('Other')
            else:
                sample_types.append('Unknown')
        
        self.sample_types = pd.Series(sample_types, index=self.data.columns, name='SampleType')
        
        # Count sample types
        type_counts = self.sample_types.value_counts()
        self.logger.info("Sample type distribution:")
        for sample_type, count in type_counts.items():
            self.logger.info(f"  - {sample_type}: {count}")
        
        return self.sample_types
    
    def filter_tumor_samples(self) -> pd.DataFrame:
        """
        Filter to keep only primary tumor samples (01A).
        
        Returns:
            DataFrame with only tumor samples
        """
        if self.sample_types is None:
            self.identify_sample_types()
        
        self.logger.info("Filtering for tumor samples only...")
        
        tumor_mask = self.sample_types == 'Tumor'
        tumor_data = self.data.loc[:, tumor_mask]
        
        n_original = self.data.shape[1]
        n_filtered = tumor_data.shape[1]
        n_removed = n_original - n_filtered
        
        self.logger.info(f"Filtered samples:")
        self.logger.info(f"  - Original: {n_original}")
        self.logger.info(f"  - Tumor samples kept: {n_filtered}")
        self.logger.info(f"  - Samples removed: {n_removed}")
        
        return tumor_data
    
    def get_data_statistics(self, data: Optional[pd.DataFrame] = None) -> dict:
        """
        Get basic statistics about the data.
        
        Args:
            data: DataFrame to analyze (uses self.data if None)
            
        Returns:
            Dictionary with statistics
        """
        if data is None:
            data = self.data
        
        if data is None:
            raise ValueError("No data available. Load data first.")
        
        stats = {
            'shape': data.shape,
            'n_genes': data.shape[0],
            'n_samples': data.shape[1],
            'missing_values': {
                'total': data.isnull().sum().sum(),
                'percentage': (data.isnull().sum().sum() / (data.shape[0] * data.shape[1])) * 100,
                'genes_with_missing': data.isnull().any(axis=1).sum(),
                'samples_with_missing': data.isnull().any(axis=0).sum()
            },
            'value_range': {
                'min': float(data.min().min()),
                'max': float(data.max().max()),
                'mean': float(data.mean().mean()),
                'median': float(data.median().median()),
                'std': float(data.std().std())
            },
            'duplicates': {
                'duplicate_genes': data.index.duplicated().sum(),
                'duplicate_samples': data.columns.duplicated().sum()
            }
        }
        
        return stats
    
    def validate_data(self, data: Optional[pd.DataFrame] = None) -> bool:
        """
        Validate data integrity.
        
        Args:
            data: DataFrame to validate
            
        Returns:
            True if validation passes, False otherwise
        """
        if data is None:
            data = self.data
        
        if data is None:
            self.logger.error("No data to validate.")
            return False
        
        self.logger.info("Validating data integrity...")
        
        issues = []
        
        # Check for empty data
        if data.empty:
            issues.append("Data is empty")
        
        # Check for duplicate indices
        if data.index.duplicated().any():
            n_dups = data.index.duplicated().sum()
            issues.append(f"Found {n_dups} duplicate gene IDs")
        
        # Check for duplicate columns
        if data.columns.duplicated().any():
            n_dups = data.columns.duplicated().sum()
            issues.append(f"Found {n_dups} duplicate sample IDs")
        
        # Check for all-NaN genes
        all_nan_genes = data.isnull().all(axis=1).sum()
        if all_nan_genes > 0:
            issues.append(f"Found {all_nan_genes} genes with all NaN values")
        
        # Check for all-NaN samples
        all_nan_samples = data.isnull().all(axis=0).sum()
        if all_nan_samples > 0:
            issues.append(f"Found {all_nan_samples} samples with all NaN values")
        
        # Check for infinite values
        if np.isinf(data.values).any():
            issues.append("Data contains infinite values")
        
        # Report results
        if issues:
            self.logger.warning("Data validation issues found:")
            for issue in issues:
                self.logger.warning(f"  - {issue}")
            return False
        else:
            self.logger.info("✓ Data validation passed")
            return True


def load_tcga_luad_data(
    data_file: Optional[Path] = None,
    filter_tumor: bool = True,
    validate: bool = True
) -> Tuple[pd.DataFrame, dict]:
    """
    Convenience function to load TCGA-LUAD data.
    
    Args:
        data_file: Path to data file
        filter_tumor: Whether to filter for tumor samples only
        validate: Whether to validate data
        
    Returns:
        Tuple of (data DataFrame, statistics dict)
    """
    loader = TCGALUADLoader(data_file=data_file)
    
    # Load data
    data = loader.load_gene_expression()
    
    # Filter tumor samples if requested
    if filter_tumor:
        data = loader.filter_tumor_samples()
    
    # Validate if requested
    if validate:
        loader.validate_data(data)
    
    # Get statistics
    stats = loader.get_data_statistics(data)
    
    return data, stats


if __name__ == "__main__":
    # Test the loader
    print("Testing TCGA-LUAD Loader...")
    
    loader = TCGALUADLoader()
    
    # Load gene expression
    data = loader.load_gene_expression()
    print(f"\nLoaded data shape: {data.shape}")
    
    # Identify sample types
    sample_types = loader.identify_sample_types()
    print(f"\nSample types:\n{sample_types.value_counts()}")
    
    # Filter tumor samples
    tumor_data = loader.filter_tumor_samples()
    print(f"\nTumor data shape: {tumor_data.shape}")
    
    # Get statistics
    stats = loader.get_data_statistics(tumor_data)
    print(f"\nData statistics:")
    print(f"  - Shape: {stats['shape']}")
    print(f"  - Missing values: {stats['missing_values']['percentage']:.4f}%")
    print(f"  - Value range: [{stats['value_range']['min']:.2f}, {stats['value_range']['max']:.2f}]")
    
    # Validate
    is_valid = loader.validate_data(tumor_data)
    print(f"\nValidation: {'PASSED' if is_valid else 'FAILED'}")
    
    print("\n✓ Loader test complete!")