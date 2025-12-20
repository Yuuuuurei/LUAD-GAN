"""
Clustering Algorithms Module for GAN-LUAD Clustering Project
Phase 3 & 7: Baseline and GAN-Assisted Clustering

This module provides implementations of various clustering algorithms:
- K-Means clustering
- Hierarchical clustering (Agglomerative)
- Spectral clustering
- DBSCAN (density-based, optional)

Supports both baseline (real data only) and GAN-assisted (augmented data) clustering.

Author: GAN-LUAD Team
Date: 2025
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import logging
from pathlib import Path
import json

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ClusteringPipeline:
    """
    Comprehensive clustering pipeline supporting multiple algorithms.
    """
    
    def __init__(
        self,
        data: np.ndarray,
        sample_labels: Optional[np.ndarray] = None,
        scale_data: bool = True,
        random_state: int = 42
    ):
        """
        Initialize clustering pipeline.
        
        Args:
            data: Feature matrix (n_samples, n_features)
            sample_labels: Optional labels (0=real, 1=synthetic) for tracking
            scale_data: Whether to standardize features
            random_state: Random seed for reproducibility
        """
        self.data = data
        self.sample_labels = sample_labels
        self.random_state = random_state
        
        # Standardize data if requested
        if scale_data:
            self.scaler = StandardScaler()
            self.data_scaled = self.scaler.fit_transform(data)
            logger.info("Data standardized (mean=0, std=1)")
        else:
            self.scaler = None
            self.data_scaled = data
        
        self.n_samples, self.n_features = data.shape
        
        logger.info(f"ClusteringPipeline initialized")
        logger.info(f"  Samples: {self.n_samples}")
        logger.info(f"  Features: {self.n_features}")
        if sample_labels is not None:
            n_real = (sample_labels == 0).sum()
            n_synthetic = (sample_labels == 1).sum()
            logger.info(f"  Real samples: {n_real}")
            logger.info(f"  Synthetic samples: {n_synthetic}")
    
    def kmeans_clustering(
        self,
        n_clusters: int = 3,
        n_init: int = 10,
        max_iter: int = 300,
        **kwargs
    ) -> Dict:
        """
        Perform K-Means clustering.
        
        Args:
            n_clusters: Number of clusters
            n_init: Number of initializations
            max_iter: Maximum iterations
            **kwargs: Additional KMeans parameters
            
        Returns:
            Dictionary with cluster assignments and model info
        """
        logger.info(f"Running K-Means clustering (k={n_clusters})...")
        
        model = KMeans(
            n_clusters=n_clusters,
            n_init=n_init,
            max_iter=max_iter,
            random_state=self.random_state,
            **kwargs
        )
        
        cluster_labels = model.fit_predict(self.data_scaled)
        
        result = {
            'algorithm': 'kmeans',
            'n_clusters': n_clusters,
            'labels': cluster_labels,
            'inertia': float(model.inertia_),
            'n_iter': int(model.n_iter_),
            'cluster_centers': model.cluster_centers_,
            'model': model
        }
        
        # Count samples per cluster
        unique, counts = np.unique(cluster_labels, return_counts=True)
        result['cluster_sizes'] = dict(zip(unique.tolist(), counts.tolist()))
        
        logger.info(f"✓ K-Means complete")
        logger.info(f"  Inertia: {model.inertia_:.2f}")
        logger.info(f"  Iterations: {model.n_iter_}")
        logger.info(f"  Cluster sizes: {result['cluster_sizes']}")
        
        return result
    
    def hierarchical_clustering(
        self,
        n_clusters: int = 3,
        linkage: str = 'ward',
        **kwargs
    ) -> Dict:
        """
        Perform Hierarchical (Agglomerative) clustering.
        
        Args:
            n_clusters: Number of clusters
            linkage: Linkage criterion ('ward', 'complete', 'average', 'single')
            **kwargs: Additional AgglomerativeClustering parameters
            
        Returns:
            Dictionary with cluster assignments and model info
        """
        logger.info(f"Running Hierarchical clustering (k={n_clusters}, linkage={linkage})...")
        
        model = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage=linkage,
            **kwargs
        )
        
        cluster_labels = model.fit_predict(self.data_scaled)
        
        result = {
            'algorithm': 'hierarchical',
            'n_clusters': n_clusters,
            'linkage': linkage,
            'labels': cluster_labels,
            'n_leaves': int(model.n_leaves_),
            'n_connected_components': int(model.n_connected_components_),
            'model': model
        }
        
        # Count samples per cluster
        unique, counts = np.unique(cluster_labels, return_counts=True)
        result['cluster_sizes'] = dict(zip(unique.tolist(), counts.tolist()))
        
        logger.info(f"✓ Hierarchical clustering complete")
        logger.info(f"  Cluster sizes: {result['cluster_sizes']}")
        
        return result
    
    def spectral_clustering(
        self,
        n_clusters: int = 3,
        affinity: str = 'rbf',
        n_neighbors: int = 10,
        **kwargs
    ) -> Dict:
        """
        Perform Spectral clustering.
        
        Args:
            n_clusters: Number of clusters
            affinity: Affinity metric ('rbf', 'nearest_neighbors', 'precomputed')
            n_neighbors: Number of neighbors for nearest_neighbors affinity
            **kwargs: Additional SpectralClustering parameters
            
        Returns:
            Dictionary with cluster assignments and model info
        """
        logger.info(f"Running Spectral clustering (k={n_clusters}, affinity={affinity})...")
        
        model = SpectralClustering(
            n_clusters=n_clusters,
            affinity=affinity,
            n_neighbors=n_neighbors if affinity == 'nearest_neighbors' else None,
            random_state=self.random_state,
            **kwargs
        )
        
        cluster_labels = model.fit_predict(self.data_scaled)
        
        result = {
            'algorithm': 'spectral',
            'n_clusters': n_clusters,
            'affinity': affinity,
            'labels': cluster_labels,
            'model': model
        }
        
        # Count samples per cluster
        unique, counts = np.unique(cluster_labels, return_counts=True)
        result['cluster_sizes'] = dict(zip(unique.tolist(), counts.tolist()))
        
        logger.info(f"✓ Spectral clustering complete")
        logger.info(f"  Cluster sizes: {result['cluster_sizes']}")
        
        return result
    
    def dbscan_clustering(
        self,
        eps: float = 0.5,
        min_samples: int = 5,
        **kwargs
    ) -> Dict:
        """
        Perform DBSCAN (density-based) clustering.
        
        Args:
            eps: Maximum distance between samples
            min_samples: Minimum samples in a neighborhood
            **kwargs: Additional DBSCAN parameters
            
        Returns:
            Dictionary with cluster assignments and model info
        """
        logger.info(f"Running DBSCAN clustering (eps={eps}, min_samples={min_samples})...")
        
        model = DBSCAN(
            eps=eps,
            min_samples=min_samples,
            **kwargs
        )
        
        cluster_labels = model.fit_predict(self.data_scaled)
        
        # Count clusters (excluding noise points labeled as -1)
        unique_labels = np.unique(cluster_labels)
        n_clusters = len(unique_labels[unique_labels >= 0])
        n_noise = (cluster_labels == -1).sum()
        
        result = {
            'algorithm': 'dbscan',
            'eps': eps,
            'min_samples': min_samples,
            'n_clusters': n_clusters,
            'n_noise_points': int(n_noise),
            'labels': cluster_labels,
            'model': model
        }
        
        # Count samples per cluster
        unique, counts = np.unique(cluster_labels, return_counts=True)
        result['cluster_sizes'] = dict(zip(unique.tolist(), counts.tolist()))
        
        logger.info(f"✓ DBSCAN complete")
        logger.info(f"  Clusters found: {n_clusters}")
        logger.info(f"  Noise points: {n_noise}")
        logger.info(f"  Cluster sizes: {result['cluster_sizes']}")
        
        return result
    
    def run_multiple_k(
        self,
        algorithm: str = 'kmeans',
        k_range: List[int] = [2, 3, 4, 5, 6, 7, 8, 9, 10],
        **kwargs
    ) -> Dict[int, Dict]:
        """
        Run clustering with multiple k values.
        
        Args:
            algorithm: Clustering algorithm ('kmeans', 'hierarchical', 'spectral')
            k_range: Range of k values to test
            **kwargs: Algorithm-specific parameters
            
        Returns:
            Dictionary mapping k to clustering results
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"Running {algorithm} for k in {k_range}")
        logger.info(f"{'='*80}")
        
        results = {}
        
        for k in k_range:
            logger.info(f"\nTesting k={k}...")
            
            if algorithm == 'kmeans':
                result = self.kmeans_clustering(n_clusters=k, **kwargs)
            elif algorithm == 'hierarchical':
                result = self.hierarchical_clustering(n_clusters=k, **kwargs)
            elif algorithm == 'spectral':
                result = self.spectral_clustering(n_clusters=k, **kwargs)
            else:
                raise ValueError(f"Unknown algorithm: {algorithm}")
            
            results[k] = result
        
        logger.info(f"\n✓ Completed {len(k_range)} clustering runs")
        return results
    
    def extract_real_sample_clusters(
        self,
        cluster_labels: np.ndarray
    ) -> np.ndarray:
        """
        Extract cluster assignments for real samples only (when using augmented data).
        
        Args:
            cluster_labels: Cluster labels for all samples (real + synthetic)
            
        Returns:
            Cluster labels for real samples only
        """
        if self.sample_labels is None:
            logger.warning("No sample labels provided, returning all cluster labels")
            return cluster_labels
        
        real_mask = self.sample_labels == 0
        real_cluster_labels = cluster_labels[real_mask]
        
        logger.info(f"Extracted clusters for {len(real_cluster_labels)} real samples")
        return real_cluster_labels
    
    def apply_pca_before_clustering(
        self,
        n_components: int = 50,
        explained_variance_threshold: float = 0.95
    ) -> np.ndarray:
        """
        Apply PCA dimensionality reduction before clustering.
        
        Args:
            n_components: Number of principal components (or max components)
            explained_variance_threshold: Use enough components to explain this variance
            
        Returns:
            PCA-transformed data
        """
        logger.info(f"Applying PCA before clustering...")
        
        pca = PCA(n_components=n_components, random_state=self.random_state)
        data_pca = pca.fit_transform(self.data_scaled)
        
        # Find number of components for threshold
        cumsum_var = np.cumsum(pca.explained_variance_ratio_)
        n_components_needed = np.argmax(cumsum_var >= explained_variance_threshold) + 1
        
        logger.info(f"  Components: {n_components}")
        logger.info(f"  Variance explained: {cumsum_var[-1]*100:.2f}%")
        logger.info(f"  Components for {explained_variance_threshold*100}% variance: {n_components_needed}")
        
        # Update data
        self.data_scaled = data_pca
        self.n_features = data_pca.shape[1]
        self.pca_model = pca
        
        return data_pca


class ClusteringComparison:
    """
    Compare clustering results across different configurations.
    """
    
    def __init__(self):
        self.results = {}
    
    def add_result(
        self,
        name: str,
        result: Dict
    ):
        """
        Add a clustering result for comparison.
        
        Args:
            name: Identifier for this result (e.g., 'baseline_kmeans_k3')
            result: Clustering result dictionary
        """
        self.results[name] = result
        logger.info(f"Added result: {name}")
    
    def compare_cluster_sizes(self) -> Dict:
        """
        Compare cluster size distributions across results.
        
        Returns:
            Dictionary of cluster sizes for each result
        """
        comparison = {}
        
        for name, result in self.results.items():
            comparison[name] = result['cluster_sizes']
        
        return comparison
    
    def find_optimal_k(
        self,
        metric_name: str,
        maximize: bool = True
    ) -> Tuple[int, float]:
        """
        Find optimal k based on a metric.
        
        Args:
            metric_name: Name of metric to optimize
            maximize: Whether to maximize (True) or minimize (False)
            
        Returns:
            Optimal k and its metric value
        """
        metric_values = {}
        
        for name, result in self.results.items():
            if 'n_clusters' in result and 'metrics' in result:
                k = result['n_clusters']
                if metric_name in result['metrics']:
                    metric_values[k] = result['metrics'][metric_name]
        
        if not metric_values:
            logger.warning(f"No results with metric '{metric_name}' found")
            return None, None
        
        if maximize:
            optimal_k = max(metric_values, key=metric_values.get)
        else:
            optimal_k = min(metric_values, key=metric_values.get)
        
        optimal_value = metric_values[optimal_k]
        
        logger.info(f"Optimal k={optimal_k} for {metric_name}: {optimal_value:.4f}")
        return optimal_k, optimal_value
    
    def save_results(self, save_path: Union[str, Path]):
        """
        Save all clustering results to JSON.
        
        Args:
            save_path: Path to save results
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert numpy arrays to lists for JSON serialization
        results_serializable = {}
        for name, result in self.results.items():
            result_copy = result.copy()
            
            # Convert arrays to lists
            if 'labels' in result_copy:
                result_copy['labels'] = result_copy['labels'].tolist()
            if 'cluster_centers' in result_copy:
                result_copy['cluster_centers'] = result_copy['cluster_centers'].tolist()
            
            # Remove model objects (not serializable)
            result_copy.pop('model', None)
            
            results_serializable[name] = result_copy
        
        with open(save_path, 'w') as f:
            json.dump(results_serializable, f, indent=2)
        
        logger.info(f"Results saved to: {save_path}")


# Example usage
if __name__ == "__main__":
    print("Clustering Algorithms Module - Phase 3 & 7")
    print("="*80)
    
    print("\nExample usage:")
    print("""
    import numpy as np
    from backend.clustering.algorithms import ClusteringPipeline
    
    # Load augmented data
    aug_data = np.load('data/synthetic/augmented_add_1x.npz')
    X = aug_data['data']
    sample_labels = aug_data['labels']  # 0=real, 1=synthetic
    
    # Initialize pipeline
    pipeline = ClusteringPipeline(X, sample_labels=sample_labels)
    
    # Optional: Apply PCA
    # pipeline.apply_pca_before_clustering(n_components=50)
    
    # Run K-Means for multiple k values
    results = pipeline.run_multiple_k(
        algorithm='kmeans',
        k_range=[2, 3, 4, 5, 6, 7, 8, 9, 10]
    )
    
    # Extract real sample clusters (exclude synthetic)
    for k, result in results.items():
        real_labels = pipeline.extract_real_sample_clusters(result['labels'])
        print(f"k={k}: {len(real_labels)} real samples clustered")
    
    # Run other algorithms
    hierarchical_result = pipeline.hierarchical_clustering(n_clusters=3)
    spectral_result = pipeline.spectral_clustering(n_clusters=3)
    """)
    
    print("\n" + "="*80)
    print("Module ready for use!")