"""
Unit tests for data loading module
Tests: backend/data/loader.py
"""

import pytest
import numpy as np
import pandas as pd
import torch
from pathlib import Path
import tempfile
import os

# Adjust imports based on your actual structure
# from backend.data.loader import TCGADataLoader, load_tcga_data


class TestTCGADataLoader:
    """Test suite for TCGA data loading functionality"""
    
    @pytest.fixture
    def sample_tsv_data(self):
        """Create sample TCGA-like TSV data for testing"""
        data = {
            'Ensembl_ID': ['ENSG00000000003.15', 'ENSG00000000005.6', 'ENSG00000000419.12'],
            'TCGA-38-7271-01A': [4.99, 0.00, 7.42],  # Tumor sample
            'TCGA-55-7914-01A': [5.57, 0.13, 8.01],  # Tumor sample
            'TCGA-50-5933-11A': [6.12, 0.08, 7.88],  # Normal tissue (should be filtered)
        }
        return pd.DataFrame(data)
    
    @pytest.fixture
    def temp_tsv_file(self, sample_tsv_data):
        """Create temporary TSV file for testing"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.tsv', delete=False) as f:
            sample_tsv_data.to_csv(f, sep='\t', index=False)
            temp_path = f.name
        yield temp_path
        os.unlink(temp_path)  # Cleanup
    
    def test_load_tsv_file(self, temp_tsv_file):
        """Test loading TSV file from disk"""
        # Replace with your actual loader function
        # df = load_tcga_data(temp_tsv_file)
        df = pd.read_csv(temp_tsv_file, sep='\t')
        
        assert df is not None
        assert not df.empty
        assert 'Ensembl_ID' in df.columns
        assert df.shape[0] == 3  # 3 genes
    
    def test_filter_tumor_samples(self, sample_tsv_data):
        """Test filtering for tumor samples only (01A barcodes)"""
        # Expected: Keep only columns ending with '01A'
        tumor_cols = [col for col in sample_tsv_data.columns 
                      if col.endswith('-01A') or col == 'Ensembl_ID']
        
        filtered_df = sample_tsv_data[tumor_cols]
        
        assert 'TCGA-38-7271-01A' in filtered_df.columns
        assert 'TCGA-55-7914-01A' in filtered_df.columns
        assert 'TCGA-50-5933-11A' not in filtered_df.columns  # Normal tissue excluded
        assert filtered_df.shape[1] == 3  # Ensembl_ID + 2 tumor samples
    
    def test_filter_normal_samples(self, sample_tsv_data):
        """Test that normal tissue samples (11A) are excluded"""
        normal_cols = [col for col in sample_tsv_data.columns if col.endswith('-11A')]
        assert len(normal_cols) == 1  # Should find the normal sample
        
        # After filtering, normal samples should be removed
        tumor_only = sample_tsv_data.drop(columns=normal_cols)
        assert 'TCGA-50-5933-11A' not in tumor_only.columns
    
    def test_remove_ensembl_version_suffix(self):
        """Test removing version suffixes from Ensembl IDs"""
        ensembl_ids = ['ENSG00000000003.15', 'ENSG00000000005.6', 'ENSG00000000419.12']
        expected = ['ENSG00000000003', 'ENSG00000000005', 'ENSG00000000419']
        
        cleaned = [id.split('.')[0] for id in ensembl_ids]
        
        assert cleaned == expected
    
    def test_convert_to_tensor(self, sample_tsv_data):
        """Test conversion of DataFrame to PyTorch tensor"""
        # Remove Ensembl_ID column and convert to tensor
        numeric_data = sample_tsv_data.drop(columns=['Ensembl_ID'])
        tensor_data = torch.tensor(numeric_data.values.T, dtype=torch.float32)
        
        assert isinstance(tensor_data, torch.Tensor)
        assert tensor_data.dtype == torch.float32
        assert tensor_data.shape == (3, 3)  # (samples, genes)
    
    def test_handle_missing_values(self):
        """Test handling of missing values in data"""
        data_with_nan = pd.DataFrame({
            'gene1': [1.0, np.nan, 3.0],
            'gene2': [2.0, 2.5, np.nan],
        })
        
        # Test dropping NaN rows
        dropped = data_with_nan.dropna()
        assert dropped.shape[0] == 1  # Only first row has no NaN
        
        # Test filling NaN with zero
        filled = data_with_nan.fillna(0)
        assert not filled.isnull().any().any()
        assert filled.loc[1, 'gene1'] == 0.0
    
    def test_validate_data_shape(self, sample_tsv_data):
        """Test data shape validation"""
        numeric_cols = sample_tsv_data.select_dtypes(include=[np.number]).columns
        
        assert len(numeric_cols) > 0  # Should have numeric columns
        assert sample_tsv_data[numeric_cols].shape[0] > 0  # Should have rows
    
    def test_load_empty_file(self):
        """Test handling of empty TSV file"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.tsv', delete=False) as f:
            f.write("Ensembl_ID\n")  # Header only
            temp_path = f.name
        
        try:
            df = pd.read_csv(temp_path, sep='\t')
            assert df.empty or df.shape[0] == 0
        finally:
            os.unlink(temp_path)
    
    def test_load_nonexistent_file(self):
        """Test error handling for nonexistent file"""
        with pytest.raises(FileNotFoundError):
            pd.read_csv('nonexistent_file.tsv', sep='\t')
    
    def test_data_type_consistency(self, sample_tsv_data):
        """Test that all gene expression values are numeric"""
        numeric_cols = sample_tsv_data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            assert pd.api.types.is_numeric_dtype(sample_tsv_data[col])
    
    def test_sample_barcode_format(self, sample_tsv_data):
        """Test TCGA barcode format validation"""
        sample_cols = [col for col in sample_tsv_data.columns if col.startswith('TCGA-')]
        
        for col in sample_cols:
            # TCGA barcode format: TCGA-XX-XXXX-XXX
            parts = col.split('-')
            assert len(parts) == 4
            assert parts[0] == 'TCGA'
            assert parts[-1] in ['01A', '11A']  # Tumor or normal


class TestDataLoaderIntegration:
    """Integration tests for data loading pipeline"""
    
    @pytest.fixture
    def realistic_data(self):
        """Create more realistic TCGA-like dataset"""
        n_genes = 100
        n_samples = 20
        
        genes = [f'ENSG{str(i).zfill(11)}.{np.random.randint(1, 20)}' 
                 for i in range(n_genes)]
        samples = [f'TCGA-{np.random.randint(10,99)}-{np.random.randint(1000,9999)}-01A' 
                   for _ in range(n_samples)]
        
        data = np.random.lognormal(mean=5, sigma=2, size=(n_genes, n_samples))
        df = pd.DataFrame(data, columns=samples)
        df.insert(0, 'Ensembl_ID', genes)
        
        return df
    
    def test_full_loading_pipeline(self, realistic_data):
        """Test complete data loading workflow"""
        # 1. Filter tumor samples (already done in fixture)
        tumor_samples = [col for col in realistic_data.columns 
                        if col.endswith('-01A') or col == 'Ensembl_ID']
        df = realistic_data[tumor_samples]
        
        # 2. Remove Ensembl version suffixes
        df['Ensembl_ID'] = df['Ensembl_ID'].str.split('.').str[0]
        
        # 3. Set Ensembl_ID as index
        df = df.set_index('Ensembl_ID')
        
        # 4. Convert to tensor
        tensor = torch.tensor(df.values.T, dtype=torch.float32)
        
        # Validate final result
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape[0] == 20  # 20 samples
        assert tensor.shape[1] == 100  # 100 genes
        assert not torch.isnan(tensor).any()
    
    def test_memory_efficiency(self, realistic_data):
        """Test memory usage for large datasets"""
        import sys
        
        # Test DataFrame memory usage
        df_memory = realistic_data.memory_usage(deep=True).sum()
        
        # Convert to tensor
        numeric_df = realistic_data.select_dtypes(include=[np.number])
        tensor = torch.tensor(numeric_df.values, dtype=torch.float32)
        tensor_memory = tensor.element_size() * tensor.nelement()
        
        # Tensor should be more memory efficient or comparable
        assert tensor_memory < df_memory * 2  # Allow some overhead


if __name__ == '__main__':
    pytest.main([__file__, '-v'])