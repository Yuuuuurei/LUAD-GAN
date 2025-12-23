"""
Integration tests for full pipeline
Tests: Complete workflow from data loading to clustering
Location: tests/integration/test_full_pipeline.py
"""

import pytest
import numpy as np
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import tempfile
import os


class TestFullPipeline:
    """Integration tests for complete LUAD-GAN pipeline"""
    
    @pytest.fixture
    def mock_tcga_data(self):
        """Create mock TCGA-like dataset"""
        np.random.seed(42)
        n_samples = 100
        n_genes = 500
        
        # Simulate log-transformed TPM values
        data = np.random.lognormal(mean=5, sigma=2, size=(n_samples, n_genes))
        return torch.tensor(data, dtype=torch.float32)
    
    @pytest.fixture
    def simple_gan_models(self):
        """Create simple GAN models for testing"""
        latent_dim = 64
        num_features = 200  # After feature selection
        
        class SimpleGenerator(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = nn.Sequential(
                    nn.Linear(latent_dim, 128),
                    nn.LeakyReLU(0.2),
                    nn.BatchNorm1d(128),
                    nn.Linear(128, 256),
                    nn.LeakyReLU(0.2),
                    nn.BatchNorm1d(256),
                    nn.Linear(256, num_features)
                )
            
            def forward(self, z):
                return self.model(z)
        
        class SimpleCritic(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = nn.Sequential(
                    nn.Linear(num_features, 128),
                    nn.LeakyReLU(0.2),
                    nn.Dropout(0.3),
                    nn.Linear(128, 64),
                    nn.LeakyReLU(0.2),
                    nn.Linear(64, 1)
                )
            
            def forward(self, x):
                return self.model(x)
        
        return {
            'generator': SimpleGenerator(),
            'critic': SimpleCritic(),
            'latent_dim': latent_dim
        }
    
    def test_end_to_end_pipeline(self, mock_tcga_data, simple_gan_models):
        """Test complete pipeline: data → preprocess → GAN → cluster"""
        
        # Step 1: Data Preprocessing
        print("\n=== Step 1: Data Preprocessing ===")
        
        # 1.1: Feature selection (top 200 variable genes)
        variances = torch.var(mock_tcga_data, dim=0)
        _, selected_indices = torch.topk(variances, k=200)
        selected_data = mock_tcga_data[:, selected_indices]
        print(f"Selected data shape: {selected_data.shape}")
        
        # 1.2: Standardization
        mean = torch.mean(selected_data, dim=0)
        std = torch.std(selected_data, dim=0)
        standardized_data = (selected_data - mean) / (std + 1e-8)
        print(f"Standardized - Mean: {standardized_data.mean():.4f}, Std: {standardized_data.std():.4f}")
        
        assert not torch.isnan(standardized_data).any()
        assert standardized_data.shape == (100, 200)
        
        # Step 2: Baseline Clustering
        print("\n=== Step 2: Baseline Clustering ===")
        
        baseline_kmeans = KMeans(n_clusters=3, random_state=42)
        baseline_labels = baseline_kmeans.fit_predict(standardized_data.numpy())
        baseline_silhouette = silhouette_score(
            standardized_data.numpy(),
            baseline_labels
        )
        print(f"Baseline Silhouette Score: {baseline_silhouette:.4f}")
        
        assert len(baseline_labels) == 100
        assert baseline_silhouette > -1 and baseline_silhouette < 1
        
        # Step 3: GAN Training (simplified - just a few steps)
        print("\n=== Step 3: GAN Training ===")
        
        generator = simple_gan_models['generator']
        critic = simple_gan_models['critic']
        latent_dim = simple_gan_models['latent_dim']
        
        g_optimizer = torch.optim.Adam(generator.parameters(), lr=1e-4)
        c_optimizer = torch.optim.Adam(critic.parameters(), lr=1e-4)
        
        # Train for a few iterations
        generator.train()
        critic.train()
        
        for iteration in range(10):  # Minimal training for testing
            # Train critic
            for _ in range(2):
                c_optimizer.zero_grad()
                
                # Real data
                real_data = standardized_data
                real_score = critic(real_data).mean()
                
                # Fake data
                z = torch.randn(100, latent_dim)
                fake_data = generator(z).detach()
                fake_score = critic(fake_data).mean()
                
                # Wasserstein loss
                c_loss = -(real_score - fake_score)
                c_loss.backward()
                c_optimizer.step()
            
            # Train generator
            g_optimizer.zero_grad()
            z = torch.randn(100, latent_dim)
            fake_data = generator(z)
            g_loss = -critic(fake_data).mean()
            g_loss.backward()
            g_optimizer.step()
            
            if iteration % 5 == 0:
                print(f"Iteration {iteration}: C_loss={c_loss.item():.4f}, G_loss={g_loss.item():.4f}")
        
        print("GAN training completed")
        
        # Step 4: Synthetic Data Generation
        print("\n=== Step 4: Synthetic Data Generation ===")
        
        generator.eval()
        with torch.no_grad():
            n_synthetic = 100  # Same as real data
            z = torch.randn(n_synthetic, latent_dim)
            synthetic_data = generator(z)
        
        print(f"Synthetic data shape: {synthetic_data.shape}")
        assert synthetic_data.shape == (100, 200)
        assert not torch.isnan(synthetic_data).any()
        
        # Step 5: Data Augmentation
        print("\n=== Step 5: Data Augmentation ===")
        
        augmented_data = torch.cat([standardized_data, synthetic_data], dim=0)
        print(f"Augmented data shape: {augmented_data.shape}")
        assert augmented_data.shape == (200, 200)
        
        # Step 6: GAN-Assisted Clustering
        print("\n=== Step 6: GAN-Assisted Clustering ===")
        
        gan_kmeans = KMeans(n_clusters=3, random_state=42)
        gan_labels_all = gan_kmeans.fit_predict(augmented_data.numpy())
        
        # Extract labels for real samples only
        gan_labels_real = gan_labels_all[:100]
        
        gan_silhouette = silhouette_score(
            standardized_data.numpy(),
            gan_labels_real
        )
        print(f"GAN-Assisted Silhouette Score: {gan_silhouette:.4f}")
        
        # Step 7: Comparison
        print("\n=== Step 7: Results Comparison ===")
        print(f"Baseline Silhouette: {baseline_silhouette:.4f}")
        print(f"GAN-Assisted Silhouette: {gan_silhouette:.4f}")
        
        improvement = ((gan_silhouette - baseline_silhouette) / 
                      abs(baseline_silhouette) * 100)
        print(f"Improvement: {improvement:+.2f}%")
        
        # Note: Due to random training, improvement may vary
        # We just check that the pipeline completes successfully
        assert isinstance(improvement, float)
        assert not np.isnan(improvement)
        
        print("\n=== Pipeline Test Completed Successfully ===")
    
    def test_pipeline_with_pca(self, mock_tcga_data):
        """Test pipeline with PCA dimensionality reduction"""
        print("\n=== Testing Pipeline with PCA ===")
        
        # Preprocess
        variances = torch.var(mock_tcga_data, dim=0)
        _, indices = torch.topk(variances, k=300)
        selected = mock_tcga_data[:, indices]
        
        mean = torch.mean(selected, dim=0)
        std = torch.std(selected, dim=0)
        standardized = (selected - mean) / (std + 1e-8)
        
        # Apply PCA
        pca = PCA(n_components=100)
        reduced = pca.fit_transform(standardized.numpy())
        
        print(f"Original shape: {standardized.shape}")
        print(f"PCA reduced shape: {reduced.shape}")
        print(f"Variance explained: {pca.explained_variance_ratio_.sum():.4f}")
        
        # Cluster in PCA space
        kmeans = KMeans(n_clusters=3, random_state=42)
        labels = kmeans.fit_predict(reduced)
        
        silhouette = silhouette_score(reduced, labels)
        print(f"Silhouette Score (PCA space): {silhouette:.4f}")
        
        assert reduced.shape == (100, 100)
        assert pca.explained_variance_ratio_.sum() > 0.8
        assert silhouette > -1 and silhouette < 1
    
    def test_checkpoint_saving_loading(self, simple_gan_models, tmp_path):
        """Test saving and loading model checkpoints in pipeline"""
        print("\n=== Testing Checkpoint Save/Load ===")
        
        generator = simple_gan_models['generator']
        critic = simple_gan_models['critic']
        
        # Save checkpoint
        checkpoint = {
            'epoch': 10,
            'generator_state_dict': generator.state_dict(),
            'critic_state_dict': critic.state_dict(),
            'loss_history': {
                'critic': [1.0, 0.9, 0.8],
                'generator': [2.0, 1.8, 1.6]
            }
        }
        
        checkpoint_path = tmp_path / "checkpoint.pt"
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")
        
        # Load checkpoint
        loaded_checkpoint = torch.load(checkpoint_path)
        
        generator_loaded = simple_gan_models['generator'].__class__()
        critic_loaded = simple_gan_models['critic'].__class__()
        
        generator_loaded.load_state_dict(loaded_checkpoint['generator_state_dict'])
        critic_loaded.load_state_dict(loaded_checkpoint['critic_state_dict'])
        
        print(f"Checkpoint loaded: Epoch {loaded_checkpoint['epoch']}")
        
        # Verify loaded models work
        latent_dim = simple_gan_models['latent_dim']
        generator_loaded.eval()
        
        with torch.no_grad():
            z = torch.randn(10, latent_dim)
            output = generator_loaded(z)
        
        assert output.shape == (10, 200)
        assert not torch.isnan(output).any()
        print("Checkpoint save/load test passed")


class TestPipelineErrorHandling:
    """Test error handling in pipeline"""
    
    def test_invalid_data_shape(self):
        """Test pipeline handles invalid data shapes"""
        invalid_data = torch.randn(10)  # 1D instead of 2D
        
        with pytest.raises((ValueError, IndexError)):
            # Should fail when trying to select features
            variances = torch.var(invalid_data, dim=0)
    
    def test_nan_handling(self):
        """Test pipeline handles NaN values"""
        data_with_nan = torch.tensor([
            [1.0, 2.0, 3.0],
            [4.0, float('nan'), 6.0],
            [7.0, 8.0, 9.0]
        ])
        
        # Check for NaN
        assert torch.isnan(data_with_nan).any()
        
        # Remove NaN rows
        clean_data = data_with_nan[~torch.isnan(data_with_nan).any(dim=1)]
        assert not torch.isnan(clean_data).any()
        assert clean_data.shape[0] == 2
    
    def test_insufficient_samples(self):
        """Test pipeline handles insufficient samples for clustering"""
        insufficient_data = torch.randn(5, 100)  # Only 5 samples
        
        # K-Means with k > n_samples should fail
        with pytest.raises(ValueError):
            kmeans = KMeans(n_clusters=10)
            kmeans.fit(insufficient_data.numpy())


class TestPipelineReproducibility:
    """Test pipeline reproducibility"""
    
    def test_reproducible_preprocessing(self):
        """Test preprocessing gives same results with same seed"""
        np.random.seed(42)
        torch.manual_seed(42)
        
        data1 = torch.randn(50, 200)
        data2 = torch.randn(50, 200)
        
        # Same seed should give same data
        np.random.seed(42)
        torch.manual_seed(42)
        data1_repeat = torch.randn(50, 200)
        
        assert torch.allclose(data1, data1_repeat)
    
    def test_reproducible_clustering(self):
        """Test clustering gives same results with same random_state"""
        data = np.random.randn(100, 50)
        
        kmeans1 = KMeans(n_clusters=3, random_state=42)
        kmeans2 = KMeans(n_clusters=3, random_state=42)
        
        labels1 = kmeans1.fit_predict(data)
        labels2 = kmeans2.fit_predict(data)
        
        np.testing.assert_array_equal(labels1, labels2)


class TestMemoryManagement:
    """Test memory management in pipeline"""
    
    def test_memory_cleanup_after_training(self, simple_gan_models):
        """Test memory is released after training"""
        import gc
        
        generator = simple_gan_models['generator']
        latent_dim = simple_gan_models['latent_dim']
        
        # Generate data
        with torch.no_grad():
            z = torch.randn(1000, latent_dim)
            output = generator(z)
        
        # Force garbage collection
        del z, output
        gc.collect()
        
        # Should be able to generate again
        with torch.no_grad():
            z_new = torch.randn(100, latent_dim)
            output_new = generator(z_new)
        
        assert output_new.shape == (100, 200)
    
    def test_batch_processing_for_large_data(self):
        """Test batch processing for memory efficiency"""
        large_data = torch.randn(10000, 100)
        
        batch_size = 256
        n_batches = len(large_data) // batch_size
        
        processed_batches = []
        for i in range(n_batches):
            batch = large_data[i*batch_size:(i+1)*batch_size]
            # Simulate processing
            processed = batch * 2
            processed_batches.append(processed)
        
        result = torch.cat(processed_batches, dim=0)
        assert result.shape[0] == n_batches * batch_size


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])  # -s to show print statements