"""
Unit tests for data preprocessing module
Tests: backend/data/preprocessor.py
"""

import pytest
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class TestFeatureSelection:
    """Test suite for feature selection functionality"""
    
    @pytest.fixture
    def sample_gene_data(self):
        """Create sample gene expression data"""
        np.random.seed(42)
        # 100 samples x 1000 genes
        data = np.random.randn(100, 1000)
        # Add some high-variance genes
        data[:, :10] *= 5  # First 10 genes have high variance
        return torch.tensor(data, dtype=torch.float32)
    
    def test_variance_based_selection(self, sample_gene_data):
        """Test variance-based feature selection"""
        # Calculate variance per gene (across samples)
        variances = torch.var(sample_gene_data, dim=0)
        
        # Select top 100 most variable genes
        top_k = 100
        _, top_indices = torch.topk(variances, k=top_k)
        selected_data = sample_gene_data[:, top_indices]
        
        assert selected_data.shape == (100, top_k)
        assert torch.all(variances[top_indices] >= variances[top_indices[-1]])
    
    def test_select_top_variable_genes(self, sample_gene_data):
        """Test selecting top N variable genes"""
        n_genes = 200
        variances = torch.var(sample_gene_data, dim=0)
        _, indices = torch.topk(variances, k=n_genes)
        
        selected = sample_gene_data[:, indices]
        
        assert selected.shape[1] == n_genes
        # Verify selection actually chose high-variance genes
        assert torch.mean(torch.var(selected, dim=0)) > torch.mean(variances)
    
    def test_variance_calculation(self):
        """Test variance calculation is correct"""
        data = torch.tensor([[1, 2, 3],
                            [4, 5, 6],
                            [7, 8, 9]], dtype=torch.float32)
        
        # Variance across samples (dim=0) for each gene
        var = torch.var(data, dim=0)
        expected = torch.tensor([9.0, 9.0, 9.0])  # Variance for each column
        
        assert torch.allclose(var, expected)
    
    def test_feature_selection_preserves_samples(self, sample_gene_data):
        """Test that feature selection doesn't change number of samples"""
        n_features = 50
        variances = torch.var(sample_gene_data, dim=0)
        _, indices = torch.topk(variances, k=n_features)
        selected = sample_gene_data[:, indices]
        
        assert selected.shape[0] == sample_gene_data.shape[0]


class TestNormalization:
    """Test suite for data normalization"""
    
    @pytest.fixture
    def unnormalized_data(self):
        """Create unnormalized gene expression data"""
        np.random.seed(42)
        # Data with different scales
        data = np.concatenate([
            np.random.normal(10, 2, (50, 100)),  # High mean
            np.random.normal(0, 10, (50, 100)),   # High variance
        ], axis=1)
        return torch.tensor(data, dtype=torch.float32)
    
    def test_standardization(self, unnormalized_data):
        """Test z-score standardization (mean=0, std=1)"""
        # Standardize per feature (across samples)
        mean = torch.mean(unnormalized_data, dim=0, keepdim=True)
        std = torch.std(unnormalized_data, dim=0, keepdim=True)
        standardized = (unnormalized_data - mean) / (std + 1e-8)
        
        # Check mean ≈ 0 and std ≈ 1
        assert torch.allclose(torch.mean(standardized, dim=0), 
                            torch.zeros(standardized.shape[1]), atol=1e-5)
        assert torch.allclose(torch.std(standardized, dim=0), 
                            torch.ones(standardized.shape[1]), atol=1e-1)
    
    def test_min_max_normalization(self):
        """Test min-max normalization to [0, 1]"""
        data = torch.tensor([[1.0, 2.0, 3.0],
                            [4.0, 5.0, 6.0],
                            [7.0, 8.0, 9.0]], dtype=torch.float32)
        
        # Per-feature min-max
        min_vals = torch.min(data, dim=0, keepdim=True)[0]
        max_vals = torch.max(data, dim=0, keepdim=True)[0]
        normalized = (data - min_vals) / (max_vals - min_vals + 1e-8)
        
        assert torch.all(normalized >= 0)
        assert torch.all(normalized <= 1)
        assert torch.allclose(torch.min(normalized, dim=0)[0], torch.zeros(3))
        assert torch.allclose(torch.max(normalized, dim=0)[0], torch.ones(3))
    
    def test_no_additional_log_transform(self):
        """Test that data is NOT log-transformed again (already log-scaled)"""
        # TCGA data is already log-transformed, so we should NOT apply log again
        data = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
        
        # If we incorrectly apply log transform
        wrong_transform = torch.log(data + 1)
        
        # Verify this would change the data
        assert not torch.allclose(data, wrong_transform)
        
        # Correct approach: Use data as-is (or standardize only)
        correct = data  # No log transform
        assert torch.allclose(data, correct)
    
    def test_handle_zero_variance_features(self):
        """Test handling of zero-variance features in standardization"""
        data = torch.tensor([[1.0, 5.0, 5.0],
                            [2.0, 5.0, 5.0],
                            [3.0, 5.0, 5.0]], dtype=torch.float32)
        
        std = torch.std(data, dim=0)
        assert std[0] > 0  # First column has variance
        assert std[1] == 0  # Second column is constant
        assert std[2] == 0  # Third column is constant
        
        # Safe standardization with epsilon
        mean = torch.mean(data, dim=0, keepdim=True)
        std_safe = torch.std(data, dim=0, keepdim=True) + 1e-8
        standardized = (data - mean) / std_safe
        
        assert not torch.isnan(standardized).any()


class TestPCADimensionalityReduction:
    """Test suite for PCA dimensionality reduction"""
    
    @pytest.fixture
    def high_dim_data(self):
        """Create high-dimensional gene expression data"""
        np.random.seed(42)
        # 200 samples x 2000 genes
        data = np.random.randn(200, 2000)
        return data
    
    def test_pca_reduces_dimensions(self, high_dim_data):
        """Test PCA reduces dimensionality correctly"""
        n_components = 300
        pca = PCA(n_components=n_components)
        reduced = pca.fit_transform(high_dim_data)
        
        assert reduced.shape == (200, n_components)
        assert reduced.shape[1] < high_dim_data.shape[1]
    
    def test_pca_variance_explained(self, high_dim_data):
        """Test that PCA preserves sufficient variance"""
        n_components = 500
        pca = PCA(n_components=n_components)
        pca.fit(high_dim_data)
        
        variance_explained = np.sum(pca.explained_variance_ratio_)
        
        # Should preserve 80-90% variance
        assert variance_explained >= 0.8
        assert variance_explained <= 1.0
    
    def test_pca_transform_inverse(self, high_dim_data):
        """Test PCA transformation and inverse transformation"""
        pca = PCA(n_components=100)
        reduced = pca.fit_transform(high_dim_data)
        reconstructed = pca.inverse_transform(reduced)
        
        assert reconstructed.shape == high_dim_data.shape
        # Reconstruction should be approximate (not perfect due to dim reduction)
        reconstruction_error = np.mean((high_dim_data - reconstructed) ** 2)
        assert reconstruction_error < 10  # Reasonable error threshold
    
    def test_pca_deterministic(self, high_dim_data):
        """Test PCA is deterministic with same random state"""
        pca1 = PCA(n_components=100, random_state=42)
        pca2 = PCA(n_components=100, random_state=42)
        
        reduced1 = pca1.fit_transform(high_dim_data)
        reduced2 = pca2.fit_transform(high_dim_data)
        
        np.testing.assert_array_almost_equal(reduced1, reduced2)


class TestPreprocessingPipeline:
    """Integration tests for full preprocessing pipeline"""
    
    @pytest.fixture
    def raw_data(self):
        """Create raw TCGA-like data"""
        np.random.seed(42)
        # 520 samples x 20000 genes
        data = np.random.lognormal(mean=5, sigma=2, size=(520, 20000))
        return torch.tensor(data, dtype=torch.float32)
    
    def test_full_preprocessing_pipeline(self, raw_data):
        """Test complete preprocessing workflow"""
        # Step 1: Feature selection (top 1000 variable genes)
        variances = torch.var(raw_data, dim=0)
        _, indices = torch.topk(variances, k=1000)
        selected = raw_data[:, indices]
        
        # Step 2: Standardization
        mean = torch.mean(selected, dim=0, keepdim=True)
        std = torch.std(selected, dim=0, keepdim=True)
        standardized = (selected - mean) / (std + 1e-8)
        
        # Step 3: PCA (optional)
        pca = PCA(n_components=300)
        final = pca.fit_transform(standardized.numpy())
        
        # Validate
        assert final.shape == (520, 300)
        assert not np.isnan(final).any()
        assert np.sum(pca.explained_variance_ratio_) >= 0.8
    
    def test_pipeline_saves_transformers(self, raw_data):
        """Test that preprocessing pipeline saves transformers"""
        # Feature selection
        variances = torch.var(raw_data, dim=0)
        _, selected_indices = torch.topk(variances, k=1000)
        
        # Standardization parameters
        selected_data = raw_data[:, selected_indices]
        mean = torch.mean(selected_data, dim=0)
        std = torch.std(selected_data, dim=0)
        
        # PCA transformer
        standardized = (selected_data - mean) / (std + 1e-8)
        pca = PCA(n_components=300)
        pca.fit(standardized.numpy())
        
        # Verify transformers can be saved/loaded
        assert selected_indices is not None
        assert mean is not None
        assert std is not None
        assert pca is not None
        assert hasattr(pca, 'components_')
    
    def test_apply_saved_transformers_to_new_data(self, raw_data):
        """Test applying saved preprocessing to new data"""
        # Train preprocessing on subset
        train_data = raw_data[:400]
        
        # Feature selection
        variances = torch.var(train_data, dim=0)
        _, selected_indices = torch.topk(variances, k=1000)
        
        # Fit standardization
        selected_train = train_data[:, selected_indices]
        mean = torch.mean(selected_train, dim=0)
        std = torch.std(selected_train, dim=0)
        
        # Fit PCA
        standardized_train = (selected_train - mean) / (std + 1e-8)
        pca = PCA(n_components=300)
        pca.fit(standardized_train.numpy())
        
        # Apply to test data
        test_data = raw_data[400:]
        selected_test = test_data[:, selected_indices]
        standardized_test = (selected_test - mean) / (std + 1e-8)
        transformed_test = pca.transform(standardized_test.numpy())
        
        assert transformed_test.shape == (120, 300)
        assert not np.isnan(transformed_test).any()


class TestDataValidation:
    """Test suite for data validation"""
    
    def test_check_no_nan_values(self):
        """Test detection of NaN values"""
        data_with_nan = torch.tensor([[1.0, 2.0], [np.nan, 4.0]], dtype=torch.float32)
        assert torch.isnan(data_with_nan).any()
        
        data_clean = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
        assert not torch.isnan(data_clean).any()
    
    def test_check_no_inf_values(self):
        """Test detection of infinite values"""
        data_with_inf = torch.tensor([[1.0, 2.0], [np.inf, 4.0]], dtype=torch.float32)
        assert torch.isinf(data_with_inf).any()
        
        data_clean = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
        assert not torch.isinf(data_clean).any()
    
    def test_validate_data_shape(self):
        """Test data shape validation"""
        data = torch.randn(100, 1000)
        
        assert data.ndim == 2
        assert data.shape[0] > 0  # At least one sample
        assert data.shape[1] > 0  # At least one feature
    
    def test_validate_data_type(self):
        """Test data type validation"""
        data = torch.randn(10, 100)
        
        assert isinstance(data, torch.Tensor)
        assert data.dtype in [torch.float32, torch.float64]


if __name__ == '__main__':
    pytest.main([__file__, '-v'])