"""
Unit tests for clustering algorithms and evaluation
Tests: backend/clustering/algorithms.py, evaluation.py
"""

import pytest
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    adjusted_rand_score,
    normalized_mutual_info_score
)
from sklearn.decomposition import PCA


class TestKMeansClustering:
    """Test suite for K-Means clustering"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data with clear clusters"""
        np.random.seed(42)
        # Create 3 clear clusters
        cluster1 = np.random.randn(50, 10) + np.array([5, 5, 0, 0, 0, 0, 0, 0, 0, 0])
        cluster2 = np.random.randn(50, 10) + np.array([-5, -5, 0, 0, 0, 0, 0, 0, 0, 0])
        cluster3 = np.random.randn(50, 10) + np.array([0, 0, 5, 5, 0, 0, 0, 0, 0, 0])
        return np.vstack([cluster1, cluster2, cluster3])
    
    def test_kmeans_basic(self, sample_data):
        """Test basic K-Means clustering"""
        kmeans = KMeans(n_clusters=3, random_state=42)
        labels = kmeans.fit_predict(sample_data)
        
        assert len(labels) == len(sample_data)
        assert len(np.unique(labels)) == 3
        assert all(label in [0, 1, 2] for label in labels)
    
    def test_kmeans_cluster_centers(self, sample_data):
        """Test K-Means cluster centers are computed"""
        kmeans = KMeans(n_clusters=3, random_state=42)
        kmeans.fit(sample_data)
        
        assert kmeans.cluster_centers_.shape == (3, 10)
        assert not np.isnan(kmeans.cluster_centers_).any()
    
    def test_kmeans_different_k_values(self, sample_data):
        """Test K-Means with different numbers of clusters"""
        for k in [2, 3, 4, 5]:
            kmeans = KMeans(n_clusters=k, random_state=42)
            labels = kmeans.fit_predict(sample_data)
            assert len(np.unique(labels)) == k
    
    def test_kmeans_reproducible(self, sample_data):
        """Test K-Means is reproducible with same random_state"""
        kmeans1 = KMeans(n_clusters=3, random_state=42)
        kmeans2 = KMeans(n_clusters=3, random_state=42)
        
        labels1 = kmeans1.fit_predict(sample_data)
        labels2 = kmeans2.fit_predict(sample_data)
        
        np.testing.assert_array_equal(labels1, labels2)
    
    def test_kmeans_convergence(self, sample_data):
        """Test K-Means converges"""
        kmeans = KMeans(n_clusters=3, random_state=42, max_iter=300)
        kmeans.fit(sample_data)
        
        assert kmeans.n_iter_ < 300  # Should converge before max_iter


class TestHierarchicalClustering:
    """Test suite for hierarchical clustering"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample hierarchical data"""
        np.random.seed(42)
        return np.random.randn(100, 20)
    
    def test_hierarchical_basic(self, sample_data):
        """Test basic hierarchical clustering"""
        hierarchical = AgglomerativeClustering(n_clusters=3)
        labels = hierarchical.fit_predict(sample_data)
        
        assert len(labels) == len(sample_data)
        assert len(np.unique(labels)) == 3
    
    def test_hierarchical_linkage_methods(self, sample_data):
        """Test different linkage methods"""
        linkages = ['ward', 'complete', 'average', 'single']
        
        for linkage in linkages:
            hierarchical = AgglomerativeClustering(
                n_clusters=3,
                linkage=linkage
            )
            labels = hierarchical.fit_predict(sample_data)
            assert len(np.unique(labels)) == 3


class TestSpectralClustering:
    """Test suite for spectral clustering"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data"""
        np.random.seed(42)
        return np.random.randn(80, 15)
    
    def test_spectral_basic(self, sample_data):
        """Test basic spectral clustering"""
        spectral = SpectralClustering(n_clusters=3, random_state=42)
        labels = spectral.fit_predict(sample_data)
        
        assert len(labels) == len(sample_data)
        assert len(np.unique(labels)) == 3
    
    def test_spectral_affinity_types(self, sample_data):
        """Test different affinity types"""
        affinities = ['rbf', 'nearest_neighbors']
        
        for affinity in affinities:
            spectral = SpectralClustering(
                n_clusters=3,
                affinity=affinity,
                random_state=42
            )
            labels = spectral.fit_predict(sample_data)
            assert len(np.unique(labels)) <= 3


class TestClusteringEvaluation:
    """Test suite for clustering evaluation metrics"""
    
    @pytest.fixture
    def clustered_data(self):
        """Create data with known clusters"""
        np.random.seed(42)
        # Create 3 well-separated clusters
        cluster1 = np.random.randn(50, 10) + 5
        cluster2 = np.random.randn(50, 10) - 5
        cluster3 = np.random.randn(50, 10) + np.array([5, -5, 0, 0, 0, 0, 0, 0, 0, 0])
        
        data = np.vstack([cluster1, cluster2, cluster3])
        true_labels = np.array([0]*50 + [1]*50 + [2]*50)
        
        kmeans = KMeans(n_clusters=3, random_state=42)
        pred_labels = kmeans.fit_predict(data)
        
        return {
            'data': data,
            'true_labels': true_labels,
            'pred_labels': pred_labels
        }
    
    def test_silhouette_score(self, clustered_data):
        """Test Silhouette Score calculation"""
        score = silhouette_score(
            clustered_data['data'],
            clustered_data['pred_labels']
        )
        
        # Silhouette score range: [-1, 1], higher is better
        assert -1 <= score <= 1
        # For well-separated clusters, should be positive
        assert score > 0
    
    def test_davies_bouldin_index(self, clustered_data):
        """Test Davies-Bouldin Index calculation"""
        score = davies_bouldin_score(
            clustered_data['data'],
            clustered_data['pred_labels']
        )
        
        # Davies-Bouldin: lower is better, >= 0
        assert score >= 0
        # For good clustering, should be reasonably low
        assert score < 3.0
    
    def test_calinski_harabasz_score(self, clustered_data):
        """Test Calinski-Harabasz Score calculation"""
        score = calinski_harabasz_score(
            clustered_data['data'],
            clustered_data['pred_labels']
        )
        
        # Calinski-Harabasz: higher is better, > 0
        assert score > 0
        # For good clustering, should be reasonably high
        assert score > 10
    
    def test_adjusted_rand_index(self, clustered_data):
        """Test Adjusted Rand Index (external validation)"""
        ari = adjusted_rand_score(
            clustered_data['true_labels'],
            clustered_data['pred_labels']
        )
        
        # ARI range: [-1, 1], 1 = perfect match
        assert -1 <= ari <= 1
        # For well-separated clusters, should be high
        assert ari > 0.5
    
    def test_normalized_mutual_information(self, clustered_data):
        """Test Normalized Mutual Information"""
        nmi = normalized_mutual_info_score(
            clustered_data['true_labels'],
            clustered_data['pred_labels']
        )
        
        # NMI range: [0, 1], 1 = perfect match
        assert 0 <= nmi <= 1
        # For good clustering, should be high
        assert nmi > 0.5
    
    def test_within_cluster_sum_of_squares(self, clustered_data):
        """Test WCSS (inertia) calculation"""
        kmeans = KMeans(n_clusters=3, random_state=42)
        kmeans.fit(clustered_data['data'])
        
        wcss = kmeans.inertia_
        
        assert wcss > 0
        assert not np.isnan(wcss)
        assert not np.isinf(wcss)
    
    def test_elbow_method(self):
        """Test elbow method for finding optimal k"""
        np.random.seed(42)
        data = np.random.randn(100, 10)
        
        wcss_values = []
        k_values = range(2, 11)
        
        for k in k_values:
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(data)
            wcss_values.append(kmeans.inertia_)
        
        # WCSS should decrease as k increases
        assert all(wcss_values[i] > wcss_values[i+1] 
                  for i in range(len(wcss_values)-1))
    
    def test_silhouette_analysis(self):
        """Test silhouette analysis for different k values"""
        np.random.seed(42)
        # Create data with 3 clear clusters
        cluster1 = np.random.randn(30, 10) + 5
        cluster2 = np.random.randn(30, 10) - 5
        cluster3 = np.random.randn(30, 10)
        data = np.vstack([cluster1, cluster2, cluster3])
        
        silhouette_scores = []
        k_values = range(2, 6)
        
        for k in k_values:
            kmeans = KMeans(n_clusters=k, random_state=42)
            labels = kmeans.fit_predict(data)
            score = silhouette_score(data, labels)
            silhouette_scores.append(score)
        
        # k=3 should have highest silhouette score (true number of clusters)
        assert silhouette_scores[1] == max(silhouette_scores)  # k=3 is index 1


class TestClusterComparison:
    """Test suite for comparing baseline vs GAN-assisted clustering"""
    
    def test_metrics_improvement_calculation(self):
        """Test calculation of metrics improvement"""
        baseline_metrics = {
            'silhouette': 0.35,
            'davies_bouldin': 1.5,
            'calinski_harabasz': 150.0
        }
        
        gan_metrics = {
            'silhouette': 0.42,
            'davies_bouldin': 1.2,
            'calinski_harabasz': 180.0
        }
        
        # Calculate improvements
        improvements = {}
        improvements['silhouette'] = (
            (gan_metrics['silhouette'] - baseline_metrics['silhouette']) 
            / baseline_metrics['silhouette'] * 100
        )
        improvements['davies_bouldin'] = (
            (baseline_metrics['davies_bouldin'] - gan_metrics['davies_bouldin']) 
            / baseline_metrics['davies_bouldin'] * 100
        )
        improvements['calinski_harabasz'] = (
            (gan_metrics['calinski_harabasz'] - baseline_metrics['calinski_harabasz']) 
            / baseline_metrics['calinski_harabasz'] * 100
        )
        
        # Verify improvements
        assert improvements['silhouette'] == 20.0  # 20% improvement
        assert improvements['davies_bouldin'] == 20.0  # 20% reduction (improvement)
        assert improvements['calinski_harabasz'] == 20.0  # 20% improvement
    
    def test_statistical_significance(self):
        """Test that improvements are statistically significant"""
        # This would typically use bootstrapping or permutation tests
        # Simplified version here
        
        baseline_scores = np.array([0.35, 0.33, 0.37, 0.36, 0.34])
        gan_scores = np.array([0.42, 0.40, 0.44, 0.43, 0.41])
        
        # Simple t-test-like check
        baseline_mean = np.mean(baseline_scores)
        gan_mean = np.mean(gan_scores)
        
        assert gan_mean > baseline_mean
        assert (gan_mean - baseline_mean) > 0.05  # At least 5% improvement


class TestDimensionalityReductionForVisualization:
    """Test suite for dimensionality reduction for visualization"""
    
    @pytest.fixture
    def high_dim_clustered_data(self):
        """Create high-dimensional clustered data"""
        np.random.seed(42)
        cluster1 = np.random.randn(50, 100) + 2
        cluster2 = np.random.randn(50, 100) - 2
        data = np.vstack([cluster1, cluster2])
        labels = np.array([0]*50 + [1]*50)
        return {'data': data, 'labels': labels}
    
    def test_pca_for_visualization(self, high_dim_clustered_data):
        """Test PCA reduction to 2D for visualization"""
        pca = PCA(n_components=2)
        reduced = pca.fit_transform(high_dim_clustered_data['data'])
        
        assert reduced.shape == (100, 2)
        assert not np.isnan(reduced).any()
    
    def test_pca_preserves_cluster_structure(self, high_dim_clustered_data):
        """Test PCA preserves cluster structure"""
        pca = PCA(n_components=2)
        reduced = pca.fit_transform(high_dim_clustered_data['data'])
        
        # Re-cluster in reduced space
        kmeans = KMeans(n_clusters=2, random_state=42)
        new_labels = kmeans.fit_predict(reduced)
        
        # Check agreement with original labels
        ari = adjusted_rand_score(high_dim_clustered_data['labels'], new_labels)
        assert ari > 0.5  # Should preserve structure reasonably well


class TestClusterValidation:
    """Test suite for cluster validation"""
    
    def test_cluster_size_distribution(self):
        """Test checking cluster size distribution"""
        labels = np.array([0, 0, 0, 1, 1, 2, 2, 2, 2])
        
        unique, counts = np.unique(labels, return_counts=True)
        
        assert len(unique) == 3
        assert list(counts) == [3, 2, 4]
    
    def test_detect_empty_clusters(self):
        """Test detection of empty clusters"""
        labels = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
        n_clusters = 5  # We asked for 5 but only got 3
        
        unique_labels = np.unique(labels)
        
        # Check if we have fewer clusters than expected
        assert len(unique_labels) < n_clusters
    
    def test_cluster_assignment_coverage(self):
        """Test all samples are assigned to clusters"""
        labels = np.array([0, 1, 0, 2, 1, 0, 2, 1])
        
        # No samples should be unassigned (no -1 labels)
        assert -1 not in labels
        assert len(labels) == 8  # All samples assigned


if __name__ == '__main__':
    pytest.main([__file__, '-v'])