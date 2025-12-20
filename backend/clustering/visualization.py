"""
Clustering Visualization Module for GAN-LUAD Clustering Project
Phase 3 & 7: Visualize clustering results

This module provides visualization functions for:
- PCA/t-SNE scatter plots with cluster colors
- Silhouette plots
- Elbow curves
- Cluster size distributions
- Comparison plots (baseline vs GAN-assisted)

Author: GAN-LUAD Team
Date: 2025
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ClusterVisualizer:
    """
    Comprehensive visualization for clustering results.
    """
    
    def __init__(
        self,
        data: np.ndarray,
        cluster_labels: np.ndarray,
        sample_labels: Optional[np.ndarray] = None,
        figsize_default: Tuple[int, int] = (12, 8)
    ):
        """
        Initialize cluster visualizer.
        
        Args:
            data: Feature matrix (n_samples, n_features)
            cluster_labels: Cluster assignments
            sample_labels: Optional labels (0=real, 1=synthetic) for tracking
            figsize_default: Default figure size
        """
        self.data = data
        self.cluster_labels = cluster_labels
        self.sample_labels = sample_labels
        self.figsize_default = figsize_default
        
        self.n_samples, self.n_features = data.shape
        self.n_clusters = len(np.unique(cluster_labels[cluster_labels >= 0]))
        
        # Set color palette
        self.colors = sns.color_palette("husl", self.n_clusters)
        
        logger.info(f"ClusterVisualizer initialized")
        logger.info(f"  Samples: {self.n_samples}")
        logger.info(f"  Clusters: {self.n_clusters}")
    
    def plot_pca_clusters(
        self,
        save_path: Optional[Union[str, Path]] = None,
        title: str = "PCA Visualization of Clusters",
        show_real_synthetic: bool = False
    ):
        """
        Plot clusters in 2D PCA space.
        
        Args:
            save_path: Optional path to save figure
            title: Plot title
            show_real_synthetic: If True and sample_labels provided, use markers to distinguish real/synthetic
        """
        logger.info("Creating PCA cluster visualization...")
        
        # Compute PCA
        pca = PCA(n_components=2, random_state=42)
        data_pca = pca.fit_transform(self.data)
        
        # Create plot
        fig, ax = plt.subplots(figsize=self.figsize_default)
        
        # Plot each cluster
        for cluster_id in np.unique(self.cluster_labels):
            if cluster_id == -1:  # Noise points (DBSCAN)
                continue
            
            mask = self.cluster_labels == cluster_id
            
            if show_real_synthetic and self.sample_labels is not None:
                # Separate real and synthetic
                real_mask = mask & (self.sample_labels == 0)
                synthetic_mask = mask & (self.sample_labels == 1)
                
                # Plot real samples
                if real_mask.sum() > 0:
                    ax.scatter(
                        data_pca[real_mask, 0],
                        data_pca[real_mask, 1],
                        c=[self.colors[cluster_id]],
                        marker='o',
                        s=50,
                        alpha=0.6,
                        label=f'Cluster {cluster_id} (Real)' if cluster_id == 0 else None,
                        edgecolors='black',
                        linewidths=0.5
                    )
                
                # Plot synthetic samples
                if synthetic_mask.sum() > 0:
                    ax.scatter(
                        data_pca[synthetic_mask, 0],
                        data_pca[synthetic_mask, 1],
                        c=[self.colors[cluster_id]],
                        marker='x',
                        s=50,
                        alpha=0.6,
                        label=f'Cluster {cluster_id} (Synthetic)' if cluster_id == 0 else None
                    )
            else:
                # Plot all samples together
                ax.scatter(
                    data_pca[mask, 0],
                    data_pca[mask, 1],
                    c=[self.colors[cluster_id]],
                    marker='o',
                    s=50,
                    alpha=0.6,
                    label=f'Cluster {cluster_id}',
                    edgecolors='black',
                    linewidths=0.5
                )
        
        # Plot noise points if present
        noise_mask = self.cluster_labels == -1
        if noise_mask.sum() > 0:
            ax.scatter(
                data_pca[noise_mask, 0],
                data_pca[noise_mask, 1],
                c='gray',
                marker='x',
                s=30,
                alpha=0.3,
                label='Noise'
            )
        
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.2f}%)')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.2f}%)')
        ax.set_title(title)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"PCA plot saved to: {save_path}")
        
        plt.show()
    
    def plot_tsne_clusters(
        self,
        save_path: Optional[Union[str, Path]] = None,
        title: str = "t-SNE Visualization of Clusters",
        perplexity: int = 30,
        max_samples: int = 1000
    ):
        """
        Plot clusters in 2D t-SNE space.
        
        Args:
            save_path: Optional path to save figure
            title: Plot title
            perplexity: t-SNE perplexity parameter
            max_samples: Subsample if dataset is larger (t-SNE is slow)
        """
        logger.info("Creating t-SNE cluster visualization...")
        
        # Subsample if needed
        if self.n_samples > max_samples:
            logger.info(f"Subsampling to {max_samples} samples for t-SNE")
            indices = np.random.choice(self.n_samples, max_samples, replace=False)
            data_subset = self.data[indices]
            labels_subset = self.cluster_labels[indices]
        else:
            data_subset = self.data
            labels_subset = self.cluster_labels
        
        # Compute t-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        data_tsne = tsne.fit_transform(data_subset)
        
        # Create plot
        fig, ax = plt.subplots(figsize=self.figsize_default)
        
        # Plot each cluster
        for cluster_id in np.unique(labels_subset):
            if cluster_id == -1:  # Noise points
                continue
            
            mask = labels_subset == cluster_id
            ax.scatter(
                data_tsne[mask, 0],
                data_tsne[mask, 1],
                c=[self.colors[cluster_id]],
                marker='o',
                s=50,
                alpha=0.6,
                label=f'Cluster {cluster_id}',
                edgecolors='black',
                linewidths=0.5
            )
        
        # Plot noise points if present
        noise_mask = labels_subset == -1
        if noise_mask.sum() > 0:
            ax.scatter(
                data_tsne[noise_mask, 0],
                data_tsne[noise_mask, 1],
                c='gray',
                marker='x',
                s=30,
                alpha=0.3,
                label='Noise'
            )
        
        ax.set_xlabel('t-SNE 1')
        ax.set_ylabel('t-SNE 2')
        ax.set_title(title)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"t-SNE plot saved to: {save_path}")
        
        plt.show()
    
    def plot_silhouette(
        self,
        silhouette_values: np.ndarray,
        save_path: Optional[Union[str, Path]] = None,
        title: str = "Silhouette Analysis"
    ):
        """
        Plot silhouette analysis showing per-sample silhouette scores.
        
        Args:
            silhouette_values: Per-sample silhouette scores
            save_path: Optional path to save figure
            title: Plot title
        """
        logger.info("Creating silhouette plot...")
        
        fig, ax = plt.subplots(figsize=self.figsize_default)
        
        y_lower = 10
        
        for cluster_id in range(self.n_clusters):
            # Get silhouette values for this cluster
            mask = self.cluster_labels == cluster_id
            cluster_silhouette_values = silhouette_values[mask]
            cluster_silhouette_values.sort()
            
            size_cluster = cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster
            
            color = self.colors[cluster_id]
            ax.fill_betweenx(
                np.arange(y_lower, y_upper),
                0,
                cluster_silhouette_values,
                facecolor=color,
                edgecolor=color,
                alpha=0.7
            )
            
            # Label the silhouette plots with cluster numbers
            ax.text(-0.05, y_lower + 0.5 * size_cluster, str(cluster_id))
            
            y_lower = y_upper + 10
        
        # Average silhouette score
        avg_silhouette = np.nanmean(silhouette_values)
        ax.axvline(x=avg_silhouette, color="red", linestyle="--", 
                   label=f'Average: {avg_silhouette:.3f}')
        
        ax.set_xlabel("Silhouette Coefficient")
        ax.set_ylabel("Cluster")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Silhouette plot saved to: {save_path}")
        
        plt.show()
    
    def plot_cluster_sizes(
        self,
        save_path: Optional[Union[str, Path]] = None,
        title: str = "Cluster Size Distribution"
    ):
        """
        Plot cluster size distribution as bar chart.
        
        Args:
            save_path: Optional path to save figure
            title: Plot title
        """
        logger.info("Creating cluster size plot...")
        
        # Count samples per cluster
        unique, counts = np.unique(self.cluster_labels, return_counts=True)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create bar plot
        bars = ax.bar(unique, counts, color=self.colors[:len(unique)], alpha=0.7, edgecolor='black')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.,
                height,
                f'{int(height)}',
                ha='center',
                va='bottom'
            )
        
        ax.set_xlabel('Cluster ID')
        ax.set_ylabel('Number of Samples')
        ax.set_title(title)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Cluster size plot saved to: {save_path}")
        
        plt.show()


def plot_elbow_curve(
    k_values: List[int],
    wcss_values: List[float],
    save_path: Optional[Union[str, Path]] = None,
    title: str = "Elbow Method for Optimal k"
):
    """
    Plot elbow curve for determining optimal number of clusters.
    
    Args:
        k_values: List of k values tested
        wcss_values: Corresponding WCSS values
        save_path: Optional path to save figure
        title: Plot title
    """
    logger.info("Creating elbow curve...")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(k_values, wcss_values, 'bo-', linewidth=2, markersize=8)
    
    ax.set_xlabel('Number of Clusters (k)')
    ax.set_ylabel('Within-Cluster Sum of Squares (WCSS)')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(k_values)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Elbow curve saved to: {save_path}")
    
    plt.show()


def plot_metrics_comparison(
    baseline_metrics: Dict,
    gan_metrics: Dict,
    save_path: Optional[Union[str, Path]] = None,
    title: str = "Baseline vs GAN-Assisted Clustering"
):
    """
    Plot side-by-side comparison of clustering metrics.
    
    Args:
        baseline_metrics: Baseline clustering metrics
        gan_metrics: GAN-assisted clustering metrics
        save_path: Optional path to save figure
        title: Plot title
    """
    logger.info("Creating metrics comparison plot...")
    
    metrics_to_plot = ['silhouette_score', 'davies_bouldin_index', 'calinski_harabasz_score']
    metric_names = ['Silhouette\n(higher better)', 'Davies-Bouldin\n(lower better)', 'Calinski-Harabasz\n(higher better)']
    
    baseline_values = [baseline_metrics.get(m, 0) for m in metrics_to_plot]
    gan_values = [gan_metrics.get(m, 0) for m in metrics_to_plot]
    
    x = np.arange(len(metric_names))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    bars1 = ax.bar(x - width/2, baseline_values, width, label='Baseline', alpha=0.8, color='steelblue', edgecolor='black')
    bars2 = ax.bar(x + width/2, gan_values, width, label='GAN-Assisted', alpha=0.8, color='coral', edgecolor='black')
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.,
                height,
                f'{height:.3f}',
                ha='center',
                va='bottom',
                fontsize=9
            )
    
    ax.set_xlabel('Metric')
    ax.set_ylabel('Score')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Comparison plot saved to: {save_path}")
    
    plt.show()


def plot_improvement_bars(
    improvements: Dict[str, float],
    save_path: Optional[Union[str, Path]] = None,
    title: str = "Clustering Metric Improvements (%)"
):
    """
    Plot improvement percentages as horizontal bar chart.
    
    Args:
        improvements: Dictionary mapping metric names to improvement percentages
        save_path: Optional path to save figure
        title: Plot title
    """
    logger.info("Creating improvement bars plot...")
    
    metrics = list(improvements.keys())
    values = list(improvements.values())
    
    # Color bars based on positive/negative
    colors = ['green' if v > 0 else 'red' for v in values]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars = ax.barh(metrics, values, color=colors, alpha=0.7, edgecolor='black')
    
    # Add value labels
    for i, (bar, value) in enumerate(zip(bars, values)):
        ax.text(
            value,
            i,
            f'{value:+.2f}%',
            ha='left' if value > 0 else 'right',
            va='center',
            fontweight='bold'
        )
    
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    ax.set_xlabel('Improvement (%)')
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Improvement bars saved to: {save_path}")
    
    plt.show()


# Example usage
if __name__ == "__main__":
    print("Clustering Visualization Module - Phase 3 & 7")
    print("="*80)
    
    print("\nExample usage:")
    print("""
    from backend.clustering.visualization import ClusterVisualizer, plot_metrics_comparison
    
    # Initialize visualizer
    viz = ClusterVisualizer(X, cluster_labels, sample_labels)
    
    # Create plots
    viz.plot_pca_clusters(save_path='results/gan_assisted/pca_clusters.png')
    viz.plot_tsne_clusters(save_path='results/gan_assisted/tsne_clusters.png')
    viz.plot_cluster_sizes(save_path='results/gan_assisted/cluster_sizes.png')
    
    # Compare baseline vs GAN-assisted
    plot_metrics_comparison(
        baseline_metrics,
        gan_metrics,
        save_path='results/comparison/metrics_comparison.png'
    )
    """)
    
    print("\n" + "="*80)
    print("Module ready for use!")