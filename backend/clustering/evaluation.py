"""
Clustering evaluation metrics module.
Implements various metrics for assessing clustering quality.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import (
    silhouette_score,
    silhouette_samples,
    davies_bouldin_score,
    calinski_harabasz_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
    fowlkes_mallows_score
)
import logging


class ClusteringEvaluator:
    """
    Evaluator for clustering results.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize evaluator.
        
        Args:
            logger: Logger instance
        """
        self.logger = logger or logging.getLogger(__name__)
    
    def compute_wcss(self, data: np.ndarray, labels: np.ndarray) -> float:
        """
        Compute Within-Cluster Sum of Squares (WCSS).
        
        Args:
            data: Input data
            labels: Cluster labels
            
        Returns:
            WCSS value (lower is better)
        """
        wcss = 0.0
        unique_labels = np.unique(labels[labels >= 0])  # Exclude noise (-1)
        
        for label in unique_labels:
            cluster_points = data[labels == label]
            centroid = cluster_points.mean(axis=0)
            wcss += np.sum((cluster_points - centroid) ** 2)
        
        return wcss
    
    def compute_silhouette(
        self,
        data: np.ndarray,
        labels: np.ndarray
    ) -> Tuple[float, np.ndarray]:
        """
        Compute silhouette score.
        
        Args:
            data: Input data
            labels: Cluster labels
            
        Returns:
            Tuple of (average silhouette score, per-sample scores)
        """
        # Check if we have enough samples and clusters
        unique_labels = np.unique(labels[labels >= 0])
        n_clusters = len(unique_labels)
        
        if n_clusters < 2:
            self.logger.warning("Need at least 2 clusters for silhouette score")
            return 0.0, np.zeros(len(labels))
        
        # Filter out noise points for silhouette calculation
        mask = labels >= 0
        if not mask.any():
            return 0.0, np.zeros(len(labels))
        
        try:
            avg_score = silhouette_score(data[mask], labels[mask])
            sample_scores = silhouette_samples(data[mask], labels[mask])
            
            # Create full array with zeros for noise points
            full_sample_scores = np.zeros(len(labels))
            full_sample_scores[mask] = sample_scores
            
            return avg_score, full_sample_scores
        except Exception as e:
            self.logger.warning(f"Silhouette score computation failed: {e}")
            return 0.0, np.zeros(len(labels))
    
    def compute_davies_bouldin(
        self,
        data: np.ndarray,
        labels: np.ndarray
    ) -> float:
        """
        Compute Davies-Bouldin Index.
        
        Args:
            data: Input data
            labels: Cluster labels
            
        Returns:
            Davies-Bouldin Index (lower is better)
        """
        # Filter out noise points
        mask = labels >= 0
        if not mask.any() or len(np.unique(labels[mask])) < 2:
            return float('inf')
        
        try:
            score = davies_bouldin_score(data[mask], labels[mask])
            return score
        except Exception as e:
            self.logger.warning(f"Davies-Bouldin score computation failed: {e}")
            return float('inf')
    
    def compute_calinski_harabasz(
        self,
        data: np.ndarray,
        labels: np.ndarray
    ) -> float:
        """
        Compute Calinski-Harabasz Score.
        
        Args:
            data: Input data
            labels: Cluster labels
            
        Returns:
            Calinski-Harabasz Score (higher is better)
        """
        # Filter out noise points
        mask = labels >= 0
        if not mask.any() or len(np.unique(labels[mask])) < 2:
            return 0.0
        
        try:
            score = calinski_harabasz_score(data[mask], labels[mask])
            return score
        except Exception as e:
            self.logger.warning(f"Calinski-Harabasz score computation failed: {e}")
            return 0.0
    
    def compute_all_metrics(
        self,
        data: np.ndarray,
        labels: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute all unsupervised metrics.
        
        Args:
            data: Input data
            labels: Cluster labels
            
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        # Number of clusters (excluding noise)
        unique_labels = np.unique(labels[labels >= 0])
        metrics['n_clusters'] = len(unique_labels)
        
        # Number of noise points (for DBSCAN)
        metrics['n_noise'] = int(np.sum(labels == -1))
        
        # WCSS
        metrics['wcss'] = self.compute_wcss(data, labels)
        
        # Silhouette score
        silhouette_avg, _ = self.compute_silhouette(data, labels)
        metrics['silhouette_score'] = silhouette_avg
        
        # Davies-Bouldin Index
        metrics['davies_bouldin_index'] = self.compute_davies_bouldin(data, labels)
        
        # Calinski-Harabasz Score
        metrics['calinski_harabasz_score'] = self.compute_calinski_harabasz(data, labels)
        
        return metrics
    
    def compute_external_metrics(
        self,
        labels_true: np.ndarray,
        labels_pred: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute external validation metrics (requires ground truth).
        
        Args:
            labels_true: True labels
            labels_pred: Predicted labels
            
        Returns:
            Dictionary of external metrics
        """
        metrics = {}
        
        try:
            # Adjusted Rand Index
            metrics['adjusted_rand_index'] = adjusted_rand_score(labels_true, labels_pred)
            
            # Normalized Mutual Information
            metrics['normalized_mutual_info'] = normalized_mutual_info_score(
                labels_true, labels_pred, average_method='arithmetic'
            )
            
            # Fowlkes-Mallows Score
            metrics['fowlkes_mallows_score'] = fowlkes_mallows_score(labels_true, labels_pred)
            
        except Exception as e:
            self.logger.warning(f"External metrics computation failed: {e}")
        
        return metrics
    
    def compare_clustering_results(
        self,
        results: Dict[str, Dict[str, float]]
    ) -> Dict[str, Dict]:
        """
        Compare multiple clustering results.
        
        Args:
            results: Dictionary mapping model name to metrics
            
        Returns:
            Dictionary with comparison results and rankings
        """
        if not results:
            return {}
        
        comparison = {
            'results': results,
            'rankings': {},
            'best': {}
        }
        
        # Metrics where higher is better
        maximize_metrics = [
            'silhouette_score',
            'calinski_harabasz_score',
            'adjusted_rand_index',
            'normalized_mutual_info',
            'fowlkes_mallows_score'
        ]
        
        # Metrics where lower is better
        minimize_metrics = [
            'wcss',
            'davies_bouldin_index'
        ]
        
        # Rank by each metric
        for metric in maximize_metrics + minimize_metrics:
            metric_values = {}
            
            for model_name, metrics in results.items():
                if metric in metrics:
                    value = metrics[metric]
                    # Skip invalid values
                    if not (np.isnan(value) or np.isinf(value)):
                        metric_values[model_name] = value
            
            if not metric_values:
                continue
            
            # Sort and rank
            if metric in maximize_metrics:
                sorted_models = sorted(metric_values.items(), key=lambda x: x[1], reverse=True)
            else:
                sorted_models = sorted(metric_values.items(), key=lambda x: x[1])
            
            comparison['rankings'][metric] = [model for model, _ in sorted_models]
            comparison['best'][metric] = sorted_models[0][0] if sorted_models else None
        
        return comparison
    
    def elbow_analysis(
        self,
        k_range: List[int],
        wcss_values: List[float]
    ) -> int:
        """
        Find optimal k using elbow method.
        
        Args:
            k_range: List of k values tested
            wcss_values: Corresponding WCSS values
            
        Returns:
            Optimal k value
        """
        if len(k_range) < 3:
            return k_range[0] if k_range else 2
        
        # Calculate rate of decrease
        decreases = []
        for i in range(1, len(wcss_values)):
            decrease = wcss_values[i-1] - wcss_values[i]
            decreases.append(decrease)
        
        # Find elbow (maximum decrease in rate)
        if len(decreases) < 2:
            return k_range[0]
        
        rate_changes = []
        for i in range(1, len(decreases)):
            rate_change = decreases[i-1] - decreases[i]
            rate_changes.append(rate_change)
        
        if not rate_changes:
            return k_range[0]
        
        elbow_idx = np.argmax(rate_changes) + 2  # +2 because of two derivatives
        optimal_k = k_range[min(elbow_idx, len(k_range)-1)]
        
        self.logger.info(f"Elbow method suggests k={optimal_k}")
        
        return optimal_k
    
    def silhouette_analysis(
        self,
        results: Dict[int, Tuple[float, np.ndarray]]
    ) -> int:
        """
        Find optimal k using silhouette analysis.
        
        Args:
            results: Dictionary mapping k to (avg_score, sample_scores)
            
        Returns:
            Optimal k value
        """
        if not results:
            return 2
        
        # Find k with highest average silhouette score
        best_k = max(results.items(), key=lambda x: x[1][0])[0]
        
        self.logger.info(f"Silhouette analysis suggests k={best_k}")
        
        return best_k


def print_metrics_table(results: Dict[str, Dict[str, float]]):
    """
    Print a formatted table of clustering metrics.
    
    Args:
        results: Dictionary mapping model name to metrics
    """
    if not results:
        print("No results to display")
        return
    
    # Get all metric names
    all_metrics = set()
    for metrics in results.values():
        all_metrics.update(metrics.keys())
    
    all_metrics = sorted(all_metrics)
    
    # Print header
    print("\n" + "=" * 100)
    print(f"{'Model':<30}", end="")
    for metric in all_metrics:
        print(f"{metric:>15}", end="")
    print()
    print("=" * 100)
    
    # Print rows
    for model_name, metrics in results.items():
        print(f"{model_name:<30}", end="")
        for metric in all_metrics:
            value = metrics.get(metric, float('nan'))
            if isinstance(value, float):
                print(f"{value:>15.4f}", end="")
            else:
                print(f"{value:>15}", end="")
        print()
    
    print("=" * 100)


if __name__ == "__main__":
    # Test the evaluator
    print("Testing ClusteringEvaluator...")
    
    # Generate synthetic data
    np.random.seed(42)
    data = np.random.randn(100, 10)
    labels = np.random.randint(0, 3, 100)
    
    evaluator = ClusteringEvaluator()
    
    # Test metrics
    metrics = evaluator.compute_all_metrics(data, labels)
    print("\nMetrics:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")
    
    print("\n✓ ClusteringEvaluator test complete!")