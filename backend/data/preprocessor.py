"""
Data preprocessing module for TCGA-LUAD dataset.
Handles feature selection, normalization, and data transformation.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA
import pickle
import json
from typing import Tuple, Optional, Dict, List
import logging

from backend.config import PROCESSED_DATA_DIR, RANDOM_SEED
from backend.utils import set_random_seeds, save_json


class LUADPreprocessor:
    """
    Preprocessor for TCGA-LUAD gene expression data.
    """
    
    def __init__(
        self,
        n_top_genes: int = 2000,
        min_variance: float = 0.01,
        standardize: bool = True,
        apply_pca: bool = False,
        n_pca_components: int = 500,
        random_state: int = RANDOM_SEED,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize preprocessor.
        
        Args:
            n_top_genes: Number of top variable genes to keep
            min_variance: Minimum variance threshold
            standardize: Whether to standardize features
            apply_pca: Whether to apply PCA
            n_pca_components: Number of PCA components
            random_state: Random seed
            logger: Logger instance
        """
        self.n_top_genes = n_top_genes
        self.min_variance = min_variance
        self.standardize = standardize
        self.apply_pca = apply_pca
        self.n_pca_components = n_pca_components
        self.random_state = random_state
        self.logger = logger or logging.getLogger(__name__)
        
        # Set random seeds
        set_random_seeds(random_state)
        
        # Fitted components
        self.scaler = None
        self.pca = None
        self.selected_genes = None
        self.gene_variances = None
        
    def remove_ensembl_versions(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Remove version suffixes from Ensembl gene IDs.
        
        Example: ENSG00000000003.15 -> ENSG00000000003
        
        Args:
            data: DataFrame with Ensembl IDs as index
            
        Returns:
            DataFrame with cleaned gene IDs
        """
        self.logger.info("Removing Ensembl version suffixes...")
        
        original_count = len(data.index)
        
        # Remove version suffix (everything after the dot)
        cleaned_ids = data.index.str.split('.').str[0]
        data.index = cleaned_ids
        
        # Check for duplicates after version removal
        if data.index.duplicated().any():
            n_dups = data.index.duplicated().sum()
            self.logger.warning(f"Found {n_dups} duplicate gene IDs after version removal")
            self.logger.info("Keeping first occurrence of each duplicate")
            data = data[~data.index.duplicated(keep='first')]
        
        self.logger.info(f"Gene IDs cleaned: {original_count} -> {len(data.index)}")
        
        return data
    
    def handle_missing_values(
        self,
        data: pd.DataFrame,
        method: str = 'drop'
    ) -> pd.DataFrame:
        """
        Handle missing values in the data.
        
        Args:
            data: Input DataFrame
            method: Method to handle missing values ('drop', 'impute', 'keep')
            
        Returns:
            DataFrame with missing values handled
        """
        n_missing = data.isnull().sum().sum()
        
        if n_missing == 0:
            self.logger.info("No missing values found")
            return data
        
        self.logger.info(f"Handling {n_missing} missing values using method: {method}")
        
        if method == 'drop':
            # Drop genes with any missing values
            data = data.dropna(axis=0, how='any')
            self.logger.info(f"Dropped genes with missing values. New shape: {data.shape}")
        
        elif method == 'impute':
            # Impute with median
            data = data.fillna(data.median(axis=1), axis=0)
            self.logger.info("Imputed missing values with gene-wise median")
        
        elif method == 'keep':
            self.logger.info("Keeping missing values as-is")
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return data
    
    def calculate_gene_variance(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate variance for each gene across samples.
        
        Args:
            data: DataFrame with genes as rows
            
        Returns:
            Series with variance for each gene
        """
        self.logger.info("Calculating gene variances...")
        
        variances = data.var(axis=1)
        
        self.logger.info(f"Variance statistics:")
        self.logger.info(f"  - Mean: {variances.mean():.4f}")
        self.logger.info(f"  - Median: {variances.median():.4f}")
        self.logger.info(f"  - Min: {variances.min():.4f}")
        self.logger.info(f"  - Max: {variances.max():.4f}")
        self.logger.info(f"  - Genes with variance = 0: {(variances == 0).sum()}")
        
        self.gene_variances = variances
        
        return variances
    
    def select_variable_genes(
        self,
        data: pd.DataFrame,
        variances: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """
        Select top variable genes based on variance.
        
        Args:
            data: Input DataFrame
            variances: Pre-calculated variances (optional)
            
        Returns:
            DataFrame with selected genes
        """
        if variances is None:
            variances = self.calculate_gene_variance(data)
        
        self.logger.info(f"Selecting top {self.n_top_genes} variable genes...")
        
        # Filter by minimum variance
        variances_filtered = variances[variances >= self.min_variance]
        self.logger.info(f"Genes passing minimum variance threshold ({self.min_variance}): {len(variances_filtered)}")
        
        # Select top N genes
        n_select = min(self.n_top_genes, len(variances_filtered))
        top_genes = variances_filtered.nlargest(n_select).index
        
        # Filter data
        selected_data = data.loc[top_genes]
        
        # Store selected genes
        self.selected_genes = top_genes.tolist()
        
        self.logger.info(f"Selected {len(self.selected_genes)} genes")
        self.logger.info(f"Selected genes variance range: [{variances[top_genes].min():.4f}, {variances[top_genes].max():.4f}]")
        
        return selected_data
    
    def standardize_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize features to have mean=0 and std=1.
        
        Args:
            data: Input DataFrame
            
        Returns:
            Standardized DataFrame
        """
        self.logger.info("Standardizing features (z-score normalization)...")
        
        # Fit scaler on training data
        if self.scaler is None:
            self.scaler = StandardScaler()
            # Transpose: scaler expects samples as rows
            scaled_values = self.scaler.fit_transform(data.T).T
            self.logger.info("Scaler fitted and applied")
        else:
            scaled_values = self.scaler.transform(data.T).T
            self.logger.info("Applied pre-fitted scaler")
        
        # Create DataFrame with same structure
        scaled_data = pd.DataFrame(
            scaled_values,
            index=data.index,
            columns=data.columns
        )
        
        # Verify standardization
        mean_check = scaled_data.mean(axis=1).mean()
        std_check = scaled_data.std(axis=1).mean()
        self.logger.info(f"Post-standardization check: mean={mean_check:.4f}, std={std_check:.4f}")
        
        return scaled_data
    
    def apply_pca_reduction(
        self,
        data: pd.DataFrame,
        n_components: Optional[int] = None
    ) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Apply PCA dimensionality reduction.
        
        Args:
            data: Input DataFrame (genes × samples)
            n_components: Number of components (uses self.n_pca_components if None)
            
        Returns:
            Tuple of (transformed data, loadings DataFrame)
        """
        if n_components is None:
            n_components = self.n_pca_components
        
        self.logger.info(f"Applying PCA reduction to {n_components} components...")
        
        # PCA expects samples as rows, features as columns
        data_transposed = data.T
        
        # Fit PCA
        if self.pca is None:
            self.pca = PCA(n_components=n_components, random_state=self.random_state)
            transformed = self.pca.fit_transform(data_transposed)
            self.logger.info("PCA fitted and applied")
        else:
            transformed = self.pca.transform(data_transposed)
            self.logger.info("Applied pre-fitted PCA")
        
        # Log explained variance
        explained_var = self.pca.explained_variance_ratio_
        cumsum_var = np.cumsum(explained_var)
        
        self.logger.info(f"PCA variance explained:")
        self.logger.info(f"  - First component: {explained_var[0]:.4f}")
        self.logger.info(f"  - First 10 components: {cumsum_var[9]:.4f}")
        self.logger.info(f"  - All {n_components} components: {cumsum_var[-1]:.4f}")
        
        # Create loadings DataFrame
        loadings = pd.DataFrame(
            self.pca.components_.T,
            index=data.index,
            columns=[f'PC{i+1}' for i in range(n_components)]
        )
        
        return transformed, loadings
    
    def fit_transform(
        self,
        data: pd.DataFrame,
        remove_versions: bool = True,
        handle_missing: str = 'drop'
    ) -> Tuple[np.ndarray, Dict]:
        """
        Complete preprocessing pipeline: fit and transform.
        
        Args:
            data: Input DataFrame (genes × samples)
            remove_versions: Whether to remove Ensembl version suffixes
            handle_missing: How to handle missing values
            
        Returns:
            Tuple of (processed data array, metadata dict)
        """
        self.logger.info("=" * 70)
        self.logger.info("Starting preprocessing pipeline...")
        self.logger.info("=" * 70)
        
        metadata = {
            'original_shape': data.shape,
            'original_n_genes': data.shape[0],
            'original_n_samples': data.shape[1]
        }
        
        # Step 1: Remove Ensembl versions
        if remove_versions:
            data = self.remove_ensembl_versions(data)
        
        # Step 2: Handle missing values
        data = self.handle_missing_values(data, method=handle_missing)
        metadata['after_missing_shape'] = data.shape
        
        # Step 3: Calculate variances and select genes
        variances = self.calculate_gene_variance(data)
        data = self.select_variable_genes(data, variances)
        metadata['after_selection_shape'] = data.shape
        metadata['selected_n_genes'] = len(self.selected_genes)
        
        # Step 4: Standardize
        if self.standardize:
            data = self.standardize_features(data)
        
        # Step 5: PCA (optional)
        if self.apply_pca:
            data_array, loadings = self.apply_pca_reduction(data)
            metadata['pca_applied'] = True
            metadata['n_pca_components'] = self.n_pca_components
            metadata['pca_variance_explained'] = float(np.sum(self.pca.explained_variance_ratio_))
        else:
            # Convert to numpy array (samples × features for PyTorch)
            data_array = data.T.values
            metadata['pca_applied'] = False
        
        metadata['final_shape'] = data_array.shape
        
        self.logger.info("=" * 70)
        self.logger.info("Preprocessing complete!")
        self.logger.info(f"Final shape: {data_array.shape} (samples × features)")
        self.logger.info("=" * 70)
        
        return data_array, metadata
    
    def transform(self, data: pd.DataFrame) -> np.ndarray:
        """
        Transform new data using fitted preprocessor.
        
        Args:
            data: Input DataFrame
            
        Returns:
            Transformed data array
        """
        if self.selected_genes is None:
            raise ValueError("Preprocessor not fitted. Call fit_transform() first.")
        
        # Filter to selected genes
        data = data.loc[self.selected_genes]
        
        # Standardize if needed
        if self.standardize and self.scaler is not None:
            data = pd.DataFrame(
                self.scaler.transform(data.T).T,
                index=data.index,
                columns=data.columns
            )
        
        # PCA if needed
        if self.apply_pca and self.pca is not None:
            data_array = self.pca.transform(data.T)
        else:
            data_array = data.T.values
        
        return data_array
    
    def save(self, output_dir: Path):
        """
        Save preprocessor components.
        
        Args:
            output_dir: Directory to save components
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Saving preprocessor to: {output_dir}")
        
        # Save selected genes
        if self.selected_genes is not None:
            genes_file = output_dir / "feature_names.txt"
            with open(genes_file, 'w') as f:
                f.write('\n'.join(self.selected_genes))
            self.logger.info(f"Saved selected genes to: {genes_file}")
        
        # Save scaler
        if self.scaler is not None:
            scaler_file = output_dir / "scaler.pkl"
            with open(scaler_file, 'wb') as f:
                pickle.dump(self.scaler, f)
            self.logger.info(f"Saved scaler to: {scaler_file}")
        
        # Save PCA
        if self.pca is not None:
            pca_file = output_dir / "pca_transformer.pkl"
            with open(pca_file, 'wb') as f:
                pickle.dump(self.pca, f)
            self.logger.info(f"Saved PCA transformer to: {pca_file}")
        
        # Save gene variances
        if self.gene_variances is not None:
            var_file = output_dir / "gene_variances.pkl"
            with open(var_file, 'wb') as f:
                pickle.dump(self.gene_variances, f)
            self.logger.info(f"Saved gene variances to: {var_file}")
        
        self.logger.info("Preprocessor saved successfully")
    
    def load(self, output_dir: Path):
        """
        Load preprocessor components.
        
        Args:
            output_dir: Directory containing saved components
        """
        output_dir = Path(output_dir)
        
        self.logger.info(f"Loading preprocessor from: {output_dir}")
        
        # Load selected genes
        genes_file = output_dir / "feature_names.txt"
        if genes_file.exists():
            with open(genes_file, 'r') as f:
                self.selected_genes = [line.strip() for line in f]
            self.logger.info(f"Loaded {len(self.selected_genes)} selected genes")
        
        # Load scaler
        scaler_file = output_dir / "scaler.pkl"
        if scaler_file.exists():
            with open(scaler_file, 'rb') as f:
                self.scaler = pickle.load(f)
            self.logger.info("Loaded scaler")
        
        # Load PCA
        pca_file = output_dir / "pca_transformer.pkl"
        if pca_file.exists():
            with open(pca_file, 'rb') as f:
                self.pca = pickle.load(f)
            self.logger.info("Loaded PCA transformer")
        
        # Load gene variances
        var_file = output_dir / "gene_variances.pkl"
        if var_file.exists():
            with open(var_file, 'rb') as f:
                self.gene_variances = pickle.load(f)
            self.logger.info("Loaded gene variances")
        
        self.logger.info("✓ Preprocessor loaded successfully")


if __name__ == "__main__":
    print("Testing LUADPreprocessor...")
    # Add test code here if needed