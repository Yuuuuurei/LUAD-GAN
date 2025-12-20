"""
Clustering visualization module.
Creates plots and visualizations for clustering results.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List, Dict, Tuple
from pathlib import Path
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics import silhouette_samples
import logging


class ClusteringVisualizer:
    """
    Visualizer for clustering results.
    """
    
    def __init__(
        self,
        output_dir: Optional[Path] = None,
        dpi: int = 300,
        figsize: Tuple[int, int] = (10, 8),
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize visualizer.
        
        Args:
            output_dir: Directory to save plots
            dpi: Plot resolution
            figsize: Default figure size
            logger: Logger instance
        """
        self.output_dir = Path(output_dir) if output_dir else None
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.dpi = dpi
        self.figsize = figsize
        self.logger = logger or logging.getLogger(__name__)
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.dpi'] = dpi
    
    def plot_2d_scatter(
        self,
        embedding: np.ndarray,
        labels: np.ndarray,
        title: str = "Cluster Visualization",
        xlabel: str = "Dimension 1",
        ylabel: str = "Dimension 2",
        filename: Optional[str] = None,
        show: bool = True
    ):
        """
        Plot 2D scatter plot of clusters.
        
        Args:
            embedding: 2D embedding (samples × 2)
            labels: Cluster labels
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            filename: Filename to save (optional)
            show: Whether to display plot
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Get unique labels and colors
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels[unique_labels >= 0])
        colors = plt.cm.tab10(np.linspace(0, 1, max(n_clusters, 10)))
        
        # Plot each cluster
        for i, label in enumerate(unique_labels):
            if label == -1:
                # Noise points (for DBSCAN)
                color = 'gray'
                marker = 'x'
                label_name = 'Noise'
            else:
                color = colors[label % len(colors)]
                marker = 'o'
                label_name = f'Cluster {label}'
            
            mask = labels == label
            ax.scatter(
                embedding[mask, 0],
                embedding[mask, 1],
                c=[color],
                marker=marker,
                s=50,
                alpha=0.6,
                label=label_name,
                edgecolors='black',
                linewidths=0.5
            )
        
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='best', frameon=True, fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if filename and self.output_dir:
            filepath = self.output_dir / filename
            plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
            self.logger.info(f"Saved plot: {filepath}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_3d_scatter(
        self,
        embedding: np.ndarray,
        labels: np.ndarray,
        title: str = "3D Cluster Visualization",
        filename: Optional[str] = None,
        show: bool = True
    ):
        """
        Plot 3D scatter plot of clusters.
        
        Args:
            embedding: 3D embedding (samples × 3)
            labels: Cluster labels
            title: Plot title
            filename: Filename to save (optional)
            show: Whether to display plot
        """
        from mpl_toolkits.mplot3d import Axes3D
        
        fig = plt.figure(figsize=self.figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        # Get unique labels and colors
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels[unique_labels >= 0])
        colors = plt.cm.tab10(np.linspace(0, 1, max(n_clusters, 10)))
        
        # Plot each cluster
        for i, label in enumerate(unique_labels):
            if label == -1:
                color = 'gray'
                marker = 'x'
                label_name = 'Noise'
            else:
                color = colors[label % len(colors)]
                marker = 'o'
                label_name = f'Cluster {label}'
            
            mask = labels == label
            ax.scatter(
                embedding[mask, 0],
                embedding[mask, 1],
                embedding[mask, 2],
                c=[color],
                marker=marker,
                s=50,
                alpha=0.6,
                label=label_name,
                edgecolors='black',
                linewidths=0.5
            )
        
        ax.set_xlabel('Dimension 1', fontsize=10)
        ax.set_ylabel('Dimension 2', fontsize=10)
        ax.set_zlabel('Dimension 3', fontsize=10)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='best', frameon=True, fontsize=8)
        
        plt.tight_layout()
        
        if filename and self.output_dir:
            filepath = self.output_dir / filename
            plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
            self.logger.info(f"Saved plot: {filepath}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_elbow_curve(
        self,
        k_values: List[int],
        wcss_values: List[float],
        optimal_k: Optional[int] = None,
        filename: Optional[str] = None,
        show: bool = True
    ):
        """
        Plot elbow curve for K-Means.
        
        Args:
            k_values: List of k values
            wcss_values: Corresponding WCSS values
            optimal_k: Optimal k to highlight (optional)
            filename: Filename to save (optional)
            show: Whether to display plot
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        ax.plot(k_values, wcss_values, 'bo-', linewidth=2, markersize=8)
        
        if optimal_k:
            # Highlight optimal k
            idx = k_values.index(optimal_k) if optimal_k in k_values else None
            if idx is not None:
                ax.plot(optimal_k, wcss_values[idx], 'ro', markersize=12, 
                       label=f'Optimal k={optimal_k}')
                ax.axvline(x=optimal_k, color='r', linestyle='--', alpha=0.5)
        
        ax.set_xlabel('Number of Clusters (k)', fontsize=12)
        ax.set_ylabel('Within-Cluster Sum of Squares (WCSS)', fontsize=12)
        ax.set_title('Elbow Method for Optimal k', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xticks(k_values)
        
        if optimal_k:
            ax.legend(fontsize=10)
        
        plt.tight_layout()
        
        if filename and self.output_dir:
            filepath = self.output_dir / filename
            plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
            self.logger.info(f"Saved plot: {filepath}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_silhouette_analysis(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        avg_score: float,
        filename: Optional[str] = None,
        show: bool = True
    ):
        """
        Plot silhouette analysis.
        
        Args:
            data: Input data
            labels: Cluster labels
            avg_score: Average silhouette score
            filename: Filename to save (optional)
            show: Whether to display plot
        """
        # Compute silhouette scores
        sample_scores = silhouette_samples(data, labels)
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        y_lower = 10
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels)
        
        colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))
        
        for i, label in enumerate(unique_labels):
            # Get scores for this cluster
            cluster_scores = sample_scores[labels == label]
            cluster_scores.sort()
            
            size = cluster_scores.shape[0]
            y_upper = y_lower + size
            
            color = colors[i]
            ax.fill_betweenx(
                np.arange(y_lower, y_upper),
                0,
                cluster_scores,
                facecolor=color,
                edgecolor=color,
                alpha=0.7
            )
            
            # Label cluster
            ax.text(-0.05, y_lower + 0.5 * size, f'Cluster {label}', fontsize=10)
            
            y_lower = y_upper + 10
        
        ax.set_xlabel('Silhouette Coefficient', fontsize=12)
        ax.set_ylabel('Cluster', fontsize=12)
        ax.set_title(f'Silhouette Analysis (avg = {avg_score:.3f})', 
                    fontsize=14, fontweight='bold')
        
        # Vertical line for average score
        ax.axvline(x=avg_score, color='red', linestyle='--', linewidth=2,
                  label=f'Average = {avg_score:.3f}')
        ax.legend(fontsize=10)
        
        ax.set_yticks([])
        ax.set_xlim([-0.1, 1])
        
        plt.tight_layout()
        
        if filename and self.output_dir:
            filepath = self.output_dir / filename
            plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
            self.logger.info(f"Saved plot: {filepath}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_dendrogram(
        self,
        data: np.ndarray,
        method: str = 'ward',
        filename: Optional[str] = None,
        show: bool = True
    ):
        """
        Plot dendrogram for hierarchical clustering.
        
        Args:
            data: Input data
            method: Linkage method
            filename: Filename to save (optional)
            show: Whether to display plot
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Compute linkage
        Z = linkage(data, method=method)
        
        # Plot dendrogram
        dendrogram(Z, ax=ax, color_threshold=0, above_threshold_color='gray')
        
        ax.set_xlabel('Sample Index', fontsize=12)
        ax.set_ylabel('Distance', fontsize=12)
        ax.set_title(f'Hierarchical Clustering Dendrogram ({method} linkage)', 
                    fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if filename and self.output_dir:
            filepath = self.output_dir / filename
            plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
            self.logger.info(f"Saved plot: {filepath}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_cluster_heatmap(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        feature_names: Optional[List[str]] = None,
        sample_ids: Optional[List[str]] = None,
        n_top_features: int = 50,
        filename: Optional[str] = None,
        show: bool = True
    ):
        """
        Plot heatmap of gene expression per cluster.
        
        Args:
            data: Input data (samples × features)
            labels: Cluster labels
            feature_names: Feature names (optional)
            sample_ids: Sample IDs (optional)
            n_top_features: Number of top features to show
            filename: Filename to save (optional)
            show: Whether to display plot
        """
        # Sort samples by cluster
        sorted_indices = np.argsort(labels)
        sorted_data = data[sorted_indices]
        sorted_labels = labels[sorted_indices]
        
        # Select top varying features
        if data.shape[1] > n_top_features:
            variances = np.var(sorted_data, axis=0)
            top_features = np.argsort(variances)[-n_top_features:]
            sorted_data = sorted_data[:, top_features]
            if feature_names:
                feature_names = [feature_names[i] for i in top_features]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Plot heatmap
        im = ax.imshow(sorted_data.T, aspect='auto', cmap='viridis', 
                      interpolation='nearest')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Expression', fontsize=12)
        
        # Add cluster boundaries
        unique_labels = np.unique(sorted_labels)
        boundaries = []
        for label in unique_labels:
            boundary = np.where(sorted_labels == label)[0][-1] + 0.5
            boundaries.append(boundary)
            ax.axvline(x=boundary, color='white', linewidth=2)
        
        # Labels
        ax.set_xlabel('Samples (sorted by cluster)', fontsize=12)
        ax.set_ylabel('Features', fontsize=12)
        ax.set_title('Gene Expression Heatmap per Cluster', fontsize=14, fontweight='bold')
        
        # Y-axis labels (features)
        if feature_names and len(feature_names) <= 50:
            ax.set_yticks(range(len(feature_names)))
            ax.set_yticklabels(feature_names, fontsize=6)
        else:
            ax.set_yticks([])
        
        ax.set_xticks([])
        
        plt.tight_layout()
        
        if filename and self.output_dir:
            filepath = self.output_dir / filename
            plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
            self.logger.info(f"Saved plot: {filepath}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_metrics_comparison(
        self,
        results: Dict[str, Dict[str, float]],
        metric_names: List[str],
        filename: Optional[str] = None,
        show: bool = True
    ):
        """
        Plot comparison of multiple clustering results.
        
        Args:
            results: Dictionary mapping model name to metrics
            metric_names: List of metric names to plot
            filename: Filename to save (optional)
            show: Whether to display plot
        """
        n_metrics = len(metric_names)
        fig, axes = plt.subplots(1, n_metrics, figsize=(6*n_metrics, 5))
        
        if n_metrics == 1:
            axes = [axes]
        
        model_names = list(results.keys())
        
        for i, metric in enumerate(metric_names):
            values = [results[model].get(metric, 0) for model in model_names]
            
            axes[i].bar(range(len(model_names)), values, color='steelblue', alpha=0.7)
            axes[i].set_xticks(range(len(model_names)))
            axes[i].set_xticklabels(model_names, rotation=45, ha='right', fontsize=8)
            axes[i].set_ylabel(metric, fontsize=10)
            axes[i].set_title(f'{metric}', fontsize=12, fontweight='bold')
            axes[i].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if filename and self.output_dir:
            filepath = self.output_dir / filename
            plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
            self.logger.info(f"Saved plot: {filepath}")
        
        if show:
            plt.show()
        else:
            plt.close()


if __name__ == "__main__":
    # Test the visualizer
    print("Testing ClusteringVisualizer...")
    
    # Generate synthetic data
    np.random.seed(42)
    embedding = np.random.randn(100, 2)
    labels = np.random.randint(0, 3, 100)
    
    visualizer = ClusteringVisualizer()
    
    # Test 2D plot
    visualizer.plot_2d_scatter(
        embedding, labels, 
        title="Test Clustering",
        show=False
    )
    print("✓ 2D scatter plot created")
    
    # Test elbow curve
    k_values = [2, 3, 4, 5, 6]
    wcss_values = [100, 80, 65, 55, 50]
    visualizer.plot_elbow_curve(k_values, wcss_values, optimal_k=3, show=False)
    print("✓ Elbow curve created")
    
    print("\n✓ ClusteringVisualizer test complete!")