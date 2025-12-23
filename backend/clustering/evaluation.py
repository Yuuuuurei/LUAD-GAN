"""
Clustering Evaluation Module for GAN-LUAD Clustering Project
Phase 3 & 7: Evaluate clustering quality

This module provides comprehensive evaluation metrics for clustering:
- Silhouette Score (higher is better)
- Davies-Bouldin Index (lower is better)
- Calinski-Harabasz Score (higher is better)
- Within-cluster sum of squares (WCSS)
- External validation metrics (ARI, NMI) if labels available

Author: GAN-LUAD Team
Date: 2025
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from sklearn.metrics import (
    silhouette_score, silhouette_samples,
    davies_bouldin_score,
    calinski_harabasz_score,
    adjusted_rand_score,
    normalized_mutual_info_score
)
import logging
from pathlib import Path
import json
import pandas as pd

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ClusteringEvaluator:
    """
    Comprehensive evaluation of clustering results.
    """
    
    def __init__(
        self,
        data: np.ndarray,
        cluster_labels: np.ndarray,
        true_labels: Optional[np.ndarray] = None
    ):
        """
        Initialize clustering evaluator.
        
        Args:
            data: Feature matrix (n_samples, n_features)
            cluster_labels: Predicted cluster assignments
            true_labels: Optional true labels for external validation
        """
        self.data = data
        self.cluster_labels = cluster_labels
        self.true_labels = true_labels
        
        self.n_samples = len(data)
        self.n_clusters = len(np.unique(cluster_labels[cluster_labels >= 0]))  # Exclude -1 (noise)
        
        logger.info(f"ClusteringEvaluator initialized")
        logger.info(f"  Samples: {self.n_samples}")
        logger.info(f"  Clusters: {self.n_clusters}")
        
        # Check for noise points (DBSCAN)
        self.has_noise = -1 in cluster_labels
        if self.has_noise:
            n_noise = (cluster_labels == -1).sum()
            logger.info(f"  Noise points: {n_noise}")
    
    def compute_all_metrics(self) -> Dict:
        """
        Compute all available clustering metrics.
        
        Returns:
            Dictionary containing all metrics
        """
        logger.info("Computing clustering metrics...")
        
        metrics = {}
        
        # Unsupervised metrics (always computed)
        if self.n_clusters > 1:
            # Filter out noise points for metrics that don't support them
            valid_mask = self.cluster_labels >= 0
            valid_data = self.data[valid_mask]
            valid_labels = self.cluster_labels[valid_mask]
            
            if len(np.unique(valid_labels)) > 1:
                metrics['silhouette_score'] = self._compute_silhouette(valid_data, valid_labels)
                metrics['davies_bouldin_index'] = self._compute_davies_bouldin(valid_data, valid_labels)
                metrics['calinski_harabasz_score'] = self._compute_calinski_harabasz(valid_data, valid_labels)
                metrics['wcss'] = self._compute_wcss(valid_data, valid_labels)
            else:
                logger.warning("Only one cluster found, skipping some metrics")
                metrics['silhouette_score'] = None
                metrics['davies_bouldin_index'] = None
                metrics['calinski_harabasz_score'] = None
                metrics['wcss'] = None
        else:
            logger.warning("Less than 2 clusters, skipping unsupervised metrics")
            metrics['silhouette_score'] = None
            metrics['davies_bouldin_index'] = None
            metrics['calinski_harabasz_score'] = None
            metrics['wcss'] = None
        
        # External validation metrics (if true labels provided)
        if self.true_labels is not None:
            metrics['adjusted_rand_index'] = self._compute_ari()
            metrics['normalized_mutual_info'] = self._compute_nmi()
        
        # Cluster statistics
        metrics['n_clusters'] = self.n_clusters
        metrics['cluster_sizes'] = self._compute_cluster_sizes()
        
        if self.has_noise:
            metrics['n_noise_points'] = int((self.cluster_labels == -1).sum())
        
        self._print_metrics_summary(metrics)
        
        return metrics
    
    def _compute_silhouette(self, data: np.ndarray, labels: np.ndarray) -> float:
        """
        Compute Silhouette Score.
        Range: [-1, 1], higher is better
        """
        try:
            score = silhouette_score(data, labels)
            logger.info(f"  Silhouette Score: {score:.4f}")
            return float(score)
        except Exception as e:
            logger.warning(f"Could not compute silhouette score: {e}")
            return None
    
    def _compute_davies_bouldin(self, data: np.ndarray, labels: np.ndarray) -> float:
        """
        Compute Davies-Bouldin Index.
        Range: [0, ∞), lower is better
        """
        try:
            score = davies_bouldin_score(data, labels)
            logger.info(f"  Davies-Bouldin Index: {score:.4f}")
            return float(score)
        except Exception as e:
            logger.warning(f"Could not compute Davies-Bouldin index: {e}")
            return None
    
    def _compute_calinski_harabasz(self, data: np.ndarray, labels: np.ndarray) -> float:
        """
        Compute Calinski-Harabasz Score (Variance Ratio Criterion).
        Range: [0, ∞), higher is better
        """
        try:
            score = calinski_harabasz_score(data, labels)
            logger.info(f"  Calinski-Harabasz Score: {score:.2f}")
            return float(score)
        except Exception as e:
            logger.warning(f"Could not compute Calinski-Harabasz score: {e}")
            return None
    
    def _compute_wcss(self, data: np.ndarray, labels: np.ndarray) -> float:
        """
        Compute Within-Cluster Sum of Squares (WCSS).
        Lower is better
        """
        try:
            wcss = 0
            for cluster_id in np.unique(labels):
                cluster_data = data[labels == cluster_id]
                centroid = cluster_data.mean(axis=0)
                wcss += np.sum((cluster_data - centroid) ** 2)
            
            logger.info(f"  WCSS: {wcss:.2f}")
            return float(wcss)
        except Exception as e:
            logger.warning(f"Could not compute WCSS: {e}")
            return None
    
    def _compute_ari(self) -> float:
        """
        Compute Adjusted Rand Index.
        Range: [-1, 1], higher is better
        """
        try:
            score = adjusted_rand_score(self.true_labels, self.cluster_labels)
            logger.info(f"  Adjusted Rand Index: {score:.4f}")
            return float(score)
        except Exception as e:
            logger.warning(f"Could not compute ARI: {e}")
            return None
    
    def _compute_nmi(self) -> float:
        """
        Compute Normalized Mutual Information.
        Range: [0, 1], higher is better
        """
        try:
            score = normalized_mutual_info_score(self.true_labels, self.cluster_labels)
            logger.info(f"  Normalized Mutual Info: {score:.4f}")
            return float(score)
        except Exception as e:
            logger.warning(f"Could not compute NMI: {e}")
            return None
    
    def _compute_cluster_sizes(self) -> Dict[int, int]:
        """Compute number of samples in each cluster."""
        unique, counts = np.unique(self.cluster_labels, return_counts=True)
        return dict(zip(unique.tolist(), counts.tolist()))
    
    def _print_metrics_summary(self, metrics: Dict):
        """Print formatted metrics summary."""
        logger.info("\n" + "="*80)
        logger.info("Clustering Metrics Summary")
        logger.info("="*80)
        
        if metrics['silhouette_score'] is not None:
            logger.info(f"Silhouette Score:        {metrics['silhouette_score']:.4f} (higher is better)")
        if metrics['davies_bouldin_index'] is not None:
            logger.info(f"Davies-Bouldin Index:    {metrics['davies_bouldin_index']:.4f} (lower is better)")
        if metrics['calinski_harabasz_score'] is not None:
            logger.info(f"Calinski-Harabasz Score: {metrics['calinski_harabasz_score']:.2f} (higher is better)")
        
        logger.info(f"\nNumber of clusters: {metrics['n_clusters']}")
        logger.info(f"Cluster sizes: {metrics['cluster_sizes']}")
        
        if 'adjusted_rand_index' in metrics:
            logger.info(f"\nExternal Validation:")
            logger.info(f"  ARI: {metrics['adjusted_rand_index']:.4f}")
            logger.info(f"  NMI: {metrics['normalized_mutual_info']:.4f}")
        
        logger.info("="*80)
    
    def compute_silhouette_per_sample(self) -> np.ndarray:
        """
        Compute silhouette score for each sample.
        
        Returns:
            Array of silhouette scores per sample
        """
        # Filter out noise points
        valid_mask = self.cluster_labels >= 0
        valid_data = self.data[valid_mask]
        valid_labels = self.cluster_labels[valid_mask]
        
        if len(np.unique(valid_labels)) > 1:
            silhouette_vals = silhouette_samples(valid_data, valid_labels)
            
            # Create full array with NaN for noise points
            full_silhouette = np.full(len(self.cluster_labels), np.nan)
            full_silhouette[valid_mask] = silhouette_vals
            
            return full_silhouette
        else:
            logger.warning("Cannot compute per-sample silhouette: less than 2 clusters")
            return np.full(len(self.cluster_labels), np.nan)
    
    def evaluate_elbow_method(self, wcss_values: Dict[int, float]) -> int:
        """
        Find elbow point in WCSS curve for optimal k.
        
        Args:
            wcss_values: Dictionary mapping k to WCSS
            
        Returns:
            Suggested optimal k
        """
        k_values = sorted(wcss_values.keys())
        wcss = [wcss_values[k] for k in k_values]
        
        # Simple elbow detection using rate of change
        if len(k_values) < 3:
            logger.warning("Not enough k values for elbow detection")
            return k_values[0] if k_values else None
        
        # Calculate second derivative (curvature)
        changes = np.diff(wcss)
        second_diff = np.diff(changes)
        
        # Find maximum curvature (most negative second derivative)
        elbow_idx = np.argmin(second_diff) + 1
        optimal_k = k_values[elbow_idx]
        
        logger.info(f"Elbow method suggests k={optimal_k}")
        return optimal_k


class ResultsComparator:
    """
    Compare baseline vs GAN-assisted clustering results.
    """
    
    def __init__(self):
        self.baseline_results = {}
        self.gan_results = {}
    
    def add_baseline_result(self, name: str, metrics: Dict):
        """Add baseline clustering result."""
        self.baseline_results[name] = metrics
        logger.info(f"Added baseline result: {name}")
    
    def add_gan_result(self, name: str, metrics: Dict):
        """Add GAN-assisted clustering result."""
        self.gan_results[name] = metrics
        logger.info(f"Added GAN-assisted result: {name}")
    
    def compute_improvements(self) -> Dict:
        """
        Compute improvement percentages: (GAN - Baseline) / Baseline * 100
        
        Returns:
            Dictionary of improvements for each metric
        """
        logger.info("\n" + "="*80)
        logger.info("Computing Improvements: GAN vs Baseline")
        logger.info("="*80)
        
        improvements = {}
        
        for name in self.baseline_results.keys():
            if name not in self.gan_results:
                continue
            
            baseline = self.baseline_results[name]
            gan = self.gan_results[name]
            
            improvement = {}
            
            # Silhouette Score (higher is better)
            if baseline.get('silhouette_score') and gan.get('silhouette_score'):
                improvement['silhouette_score'] = self._compute_improvement(
                    baseline['silhouette_score'],
                    gan['silhouette_score'],
                    higher_is_better=True
                )
            
            # Davies-Bouldin Index (lower is better)
            if baseline.get('davies_bouldin_index') and gan.get('davies_bouldin_index'):
                improvement['davies_bouldin_index'] = self._compute_improvement(
                    baseline['davies_bouldin_index'],
                    gan['davies_bouldin_index'],
                    higher_is_better=False
                )
            
            # Calinski-Harabasz Score (higher is better)
            if baseline.get('calinski_harabasz_score') and gan.get('calinski_harabasz_score'):
                improvement['calinski_harabasz_score'] = self._compute_improvement(
                    baseline['calinski_harabasz_score'],
                    gan['calinski_harabasz_score'],
                    higher_is_better=True
                )
            
            improvements[name] = improvement
            
            # Print improvement
            logger.info(f"\n{name}:")
            for metric, value in improvement.items():
                logger.info(f"  {metric}: {value:+.2f}%")
        
        return improvements
    
    def _compute_improvement(
        self,
        baseline_value: float,
        gan_value: float,
        higher_is_better: bool
    ) -> float:
        """
        Compute improvement percentage.
        
        Args:
            baseline_value: Baseline metric value
            gan_value: GAN-assisted metric value
            higher_is_better: Whether higher values are better
            
        Returns:
            Improvement percentage
        """
        if baseline_value == 0:
            return 0.0
        
        if higher_is_better:
            # For metrics where higher is better (silhouette, calinski-harabasz)
            improvement = ((gan_value - baseline_value) / abs(baseline_value)) * 100
        else:
            # For metrics where lower is better (davies-bouldin)
            improvement = ((baseline_value - gan_value) / abs(baseline_value)) * 100
        
        return improvement
    
    def create_comparison_table(self) -> pd.DataFrame:
        """
        Create comparison table for all metrics.
        
        Returns:
            DataFrame with baseline vs GAN-assisted metrics
        """
        rows = []
        
        for name in self.baseline_results.keys():
            if name not in self.gan_results:
                continue
            
            baseline = self.baseline_results[name]
            gan = self.gan_results[name]
            
            row = {
                'configuration': name,
                'baseline_silhouette': baseline.get('silhouette_score'),
                'gan_silhouette': gan.get('silhouette_score'),
                'baseline_davies_bouldin': baseline.get('davies_bouldin_index'),
                'gan_davies_bouldin': gan.get('davies_bouldin_index'),
                'baseline_calinski_harabasz': baseline.get('calinski_harabasz_score'),
                'gan_calinski_harabasz': gan.get('calinski_harabasz_score'),
            }
            
            rows.append(row)
        
        df = pd.DataFrame(rows)
        
        # Compute improvement columns
        if 'baseline_silhouette' in df.columns and 'gan_silhouette' in df.columns:
            df['silhouette_improvement_%'] = (
                (df['gan_silhouette'] - df['baseline_silhouette']) / 
                df['baseline_silhouette'].abs() * 100
            )
        
        if 'baseline_davies_bouldin' in df.columns and 'gan_davies_bouldin' in df.columns:
            df['davies_bouldin_improvement_%'] = (
                (df['baseline_davies_bouldin'] - df['gan_davies_bouldin']) / 
                df['baseline_davies_bouldin'].abs() * 100
            )
        
        if 'baseline_calinski_harabasz' in df.columns and 'gan_calinski_harabasz' in df.columns:
            df['calinski_harabasz_improvement_%'] = (
                (df['gan_calinski_harabasz'] - df['baseline_calinski_harabasz']) / 
                df['baseline_calinski_harabasz'].abs() * 100
            )
        
        return df
    
    def save_comparison(self, save_path: Union[str, Path]):
        """
        Save comparison results to CSV.
        
        Args:
            save_path: Path to save comparison table
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        df = self.create_comparison_table()
        df.to_csv(save_path, index=False)
        
        logger.info(f"Comparison table saved to: {save_path}")
    
    def print_summary(self):
        """Print summary of improvements."""
        improvements = self.compute_improvements()
        
        print("\n" + "="*80)
        print("IMPROVEMENT SUMMARY: GAN vs Baseline")
        print("="*80)
        
        for name, improvement in improvements.items():
            print(f"\n{name}:")
            for metric, value in improvement.items():
                symbol = "+" if value > 0 else ""
                print(f"  {metric}: {symbol}{value:.2f}%")
        
        print("\n" + "="*80)


def print_metrics_table(comparator: 'ResultsComparator'):
    """
    Print comparison table of clustering metrics.
    
    Args:
        comparator: ResultsComparator instance with baseline and GAN results
    """
    df = comparator.create_comparison_table()
    print("\n" + "="*80)
    print("CLUSTERING METRICS COMPARISON")
    print("="*80)
    print(df.to_string(index=False))
    print("="*80)


# Example usage
if __name__ == "__main__":
    print("Clustering Evaluation Module - Phase 3 & 7")
    print("="*80)
    
    print("\nExample usage:")
    print("""
    from backend.clustering.evaluation import ClusteringEvaluator, ResultsComparator
    
    # Evaluate clustering
    evaluator = ClusteringEvaluator(X, cluster_labels)
    metrics = evaluator.compute_all_metrics()
    
    # Compare baseline vs GAN-assisted
    comparator = ResultsComparator()
    comparator.add_baseline_result('kmeans_k3', baseline_metrics)
    comparator.add_gan_result('kmeans_k3', gan_metrics)
    
    improvements = comparator.compute_improvements()
    comparator.print_summary()
    comparator.save_comparison('results/comparison/metrics_comparison.csv')
    """)
    
    print("\n" + "="*80)
    print("Module ready for use!")