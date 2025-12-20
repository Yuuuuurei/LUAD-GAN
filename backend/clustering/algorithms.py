"""
Clustering algorithms module for TCGA-LUAD.
Implements various clustering methods for baseline and GAN-assisted clustering.
"""

import numpy as np
from typing import Tuple, Dict, List, Optional
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering, DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import logging

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False


class ClusteringPipeline:
    """
    Pipeline for applying clustering algorithms to gene expression data.
    """
    
    def __init__(
        self,
        random_state: int = 42,
        n_jobs: int = -1,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize clustering pipeline.
        
        Args:
            random_state: Random seed for reproducibility
            n_jobs: Number of parallel jobs (-1 for all cores)
            logger: Logger instance
        """
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.logger = logger or logging.getLogger(__name__)
        
        # Store fitted models
        self.models = {}
        self.labels = {}
        self.embeddings = {}
    
    def apply_pca(
        self,
        data: np.ndarray,
        n_components: int = 50
    ) -> Tuple[np.ndarray, PCA]:
        """
        Apply PCA for dimensionality reduction.
        
        Args:
            data: Input data (samples × features)
            n_components: Number of components
            
        Returns:
            Tuple of (transformed data, fitted PCA model)
        """
        self.logger.info(f"Applying PCA: {data.shape[1]} → {n_components} dimensions")
        
        pca = PCA(n_components=n_components, random_state=self.random_state)
        transformed = pca.fit_transform(data)
        
        variance_explained = pca.explained_variance_ratio_.sum()
        self.logger.info(f"PCA variance explained: {variance_explained:.4f}")
        
        self.embeddings['pca'] = transformed
        self.models['pca'] = pca
        
        return transformed, pca
    
    def apply_tsne(
        self,
        data: np.ndarray,
        n_components: int = 2,
        perplexity: float = 30.0,
        learning_rate: float = 200.0,
        n_iter: int = 1000
    ) -> np.ndarray:
        """
        Apply t-SNE for visualization.
        
        Args:
            data: Input data
            n_components: Number of dimensions (2 or 3)
            perplexity: t-SNE perplexity parameter
            learning_rate: Learning rate
            n_iter: Number of iterations
            
        Returns:
            t-SNE embedding
        """
        self.logger.info(f"Applying t-SNE: {data.shape[1]} → {n_components}D")
        
        tsne = TSNE(
            n_components=n_components,
            perplexity=perplexity,
            learning_rate=learning_rate,
            n_iter=n_iter,
            random_state=self.random_state,
            n_jobs=self.n_jobs
        )
        embedding = tsne.fit_transform(data)
        
        self.embeddings['tsne'] = embedding
        self.logger.info(f"t-SNE completed: KL divergence = {tsne.kl_divergence_:.4f}")
        
        return embedding
    
    def apply_umap(
        self,
        data: np.ndarray,
        n_components: int = 2,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        metric: str = "euclidean"
    ) -> Optional[np.ndarray]:
        """
        Apply UMAP for visualization.
        
        Args:
            data: Input data
            n_components: Number of dimensions (2 or 3)
            n_neighbors: Number of neighbors
            min_dist: Minimum distance
            metric: Distance metric
            
        Returns:
            UMAP embedding or None if UMAP not available
        """
        if not UMAP_AVAILABLE:
            self.logger.warning("UMAP not installed. Skipping UMAP embedding.")
            return None
        
        self.logger.info(f"Applying UMAP: {data.shape[1]} → {n_components}D")
        
        reducer = umap.UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric=metric,
            random_state=self.random_state
        )
        embedding = reducer.fit_transform(data)
        
        self.embeddings['umap'] = embedding
        self.logger.info("UMAP completed")
        
        return embedding
    
    def kmeans_clustering(
        self,
        data: np.ndarray,
        n_clusters: int,
        n_init: int = 10,
        max_iter: int = 300
    ) -> Tuple[np.ndarray, KMeans]:
        """
        Apply K-Means clustering.
        
        Args:
            data: Input data
            n_clusters: Number of clusters
            n_init: Number of initializations
            max_iter: Maximum iterations
            
        Returns:
            Tuple of (cluster labels, fitted model)
        """
        self.logger.info(f"Running K-Means with k={n_clusters}")
        
        kmeans = KMeans(
            n_clusters=n_clusters,
            n_init=n_init,
            max_iter=max_iter,
            random_state=self.random_state,
            n_jobs=self.n_jobs
        )
        labels = kmeans.fit_predict(data)
        
        self.logger.info(f"K-Means completed: inertia={kmeans.inertia_:.4f}")
        
        model_key = f'kmeans_k{n_clusters}'
        self.models[model_key] = kmeans
        self.labels[model_key] = labels
        
        return labels, kmeans
    
    def hierarchical_clustering(
        self,
        data: np.ndarray,
        n_clusters: int,
        linkage: str = "ward",
        metric: str = "euclidean"
    ) -> Tuple[np.ndarray, AgglomerativeClustering]:
        """
        Apply hierarchical clustering.
        
        Args:
            data: Input data
            n_clusters: Number of clusters
            linkage: Linkage method (ward, complete, average)
            metric: Distance metric
            
        Returns:
            Tuple of (cluster labels, fitted model)
        """
        self.logger.info(f"Running Hierarchical clustering: k={n_clusters}, linkage={linkage}")
        
        hierarchical = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage=linkage,
            metric=metric if linkage != 'ward' else 'euclidean'
        )
        labels = hierarchical.fit_predict(data)
        
        self.logger.info("Hierarchical clustering completed")
        
        model_key = f'hierarchical_k{n_clusters}_{linkage}'
        self.models[model_key] = hierarchical
        self.labels[model_key] = labels
        
        return labels, hierarchical
    
    def spectral_clustering(
        self,
        data: np.ndarray,
        n_clusters: int,
        affinity: str = "rbf",
        gamma: float = 1.0,
        n_neighbors: int = 10
    ) -> Tuple[np.ndarray, SpectralClustering]:
        """
        Apply spectral clustering.
        
        Args:
            data: Input data
            n_clusters: Number of clusters
            affinity: Affinity method (rbf, nearest_neighbors)
            gamma: Kernel coefficient for rbf
            n_neighbors: Number of neighbors
            
        Returns:
            Tuple of (cluster labels, fitted model)
        """
        self.logger.info(f"Running Spectral clustering: k={n_clusters}, affinity={affinity}")
        
        if affinity == "nearest_neighbors":
            spectral = SpectralClustering(
                n_clusters=n_clusters,
                affinity=affinity,
                n_neighbors=n_neighbors,
                random_state=self.random_state,
                n_jobs=self.n_jobs
            )
        else:
            spectral = SpectralClustering(
                n_clusters=n_clusters,
                affinity=affinity,
                gamma=gamma,
                random_state=self.random_state,
                n_jobs=self.n_jobs
            )
        
        labels = spectral.fit_predict(data)
        
        self.logger.info("Spectral clustering completed")
        
        model_key = f'spectral_k{n_clusters}_{affinity}'
        self.models[model_key] = spectral
        self.labels[model_key] = labels
        
        return labels, spectral
    
    def dbscan_clustering(
        self,
        data: np.ndarray,
        eps: float = 0.5,
        min_samples: int = 5,
        metric: str = "euclidean"
    ) -> Tuple[np.ndarray, DBSCAN]:
        """
        Apply DBSCAN clustering.
        
        Args:
            data: Input data
            eps: Maximum distance between samples
            min_samples: Minimum samples in neighborhood
            metric: Distance metric
            
        Returns:
            Tuple of (cluster labels, fitted model)
        """
        self.logger.info(f"Running DBSCAN: eps={eps}, min_samples={min_samples}")
        
        dbscan = DBSCAN(
            eps=eps,
            min_samples=min_samples,
            metric=metric,
            n_jobs=self.n_jobs
        )
        labels = dbscan.fit_predict(data)
        
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        
        self.logger.info(f"DBSCAN completed: {n_clusters} clusters, {n_noise} noise points")
        
        model_key = f'dbscan_eps{eps}_min{min_samples}'
        self.models[model_key] = dbscan
        self.labels[model_key] = labels
        
        return labels, dbscan
    
    def run_multiple_k(
        self,
        data: np.ndarray,
        algorithm: str,
        k_range: List[int],
        **kwargs
    ) -> Dict[int, Tuple[np.ndarray, object]]:
        """
        Run clustering algorithm for multiple k values.
        
        Args:
            data: Input data
            algorithm: Clustering algorithm (kmeans, hierarchical, spectral)
            k_range: List of k values to test
            **kwargs: Additional parameters for the algorithm
            
        Returns:
            Dictionary mapping k to (labels, model)
        """
        results = {}
        
        for k in k_range:
            self.logger.info(f"\nTesting k={k} with {algorithm}")
            
            if algorithm == "kmeans":
                labels, model = self.kmeans_clustering(data, k, **kwargs)
            elif algorithm == "hierarchical":
                labels, model = self.hierarchical_clustering(data, k, **kwargs)
            elif algorithm == "spectral":
                labels, model = self.spectral_clustering(data, k, **kwargs)
            else:
                raise ValueError(f"Unknown algorithm: {algorithm}")
            
            results[k] = (labels, model)
        
        return results
    
    def get_model(self, model_key: str):
        """Get fitted model by key."""
        return self.models.get(model_key)
    
    def get_labels(self, model_key: str) -> Optional[np.ndarray]:
        """Get cluster labels by key."""
        return self.labels.get(model_key)
    
    def get_embedding(self, embedding_type: str) -> Optional[np.ndarray]:
        """Get embedding by type (pca, tsne, umap)."""
        return self.embeddings.get(embedding_type)


if __name__ == "__main__":
    # Test the clustering pipeline
    print("Testing ClusteringPipeline...")
    
    # Generate synthetic data
    np.random.seed(42)
    data = np.random.randn(100, 50)
    
    pipeline = ClusteringPipeline(random_state=42)
    
    # Test K-Means
    labels, model = pipeline.kmeans_clustering(data, n_clusters=3)
    print(f"K-Means labels: {len(set(labels))} clusters")
    
    # Test Hierarchical
    labels, model = pipeline.hierarchical_clustering(data, n_clusters=3)
    print(f"Hierarchical labels: {len(set(labels))} clusters")
    
    # Test PCA
    pca_data, pca_model = pipeline.apply_pca(data, n_components=10)
    print(f"PCA shape: {pca_data.shape}")
    
    print("\n✓ ClusteringPipeline test complete!")