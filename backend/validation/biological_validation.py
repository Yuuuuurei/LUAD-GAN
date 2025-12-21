"""
Biological Validation Module for GAN-LUAD Clustering Project
Phase 8: Validate clusters using biological knowledge

This module provides:
- External validation (ARI, NMI with known subtypes)
- Differentially expressed gene analysis
- Gene signature identification
- Confusion matrix visualization

Author: GAN-LUAD Team
Date: 2025
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import logging
from scipy import stats
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BiologicalValidator:
    """
    Validate clustering results using biological knowledge.
    """
    
    def __init__(
        self,
        gene_expression: np.ndarray,
        cluster_labels: np.ndarray,
        gene_names: Optional[List[str]] = None,
        sample_ids: Optional[List[str]] = None
    ):
        """
        Initialize biological validator.
        
        Args:
            gene_expression: Gene expression matrix (n_samples, n_genes)
            cluster_labels: Cluster assignments for each sample
            gene_names: List of gene names
            sample_ids: List of sample IDs
        """
        self.gene_expression = gene_expression
        self.cluster_labels = cluster_labels
        self.gene_names = gene_names if gene_names else [f"Gene_{i}" for i in range(gene_expression.shape[1])]
        self.sample_ids = sample_ids if sample_ids else [f"Sample_{i}" for i in range(gene_expression.shape[0])]
        
        self.n_samples, self.n_genes = gene_expression.shape
        self.n_clusters = len(np.unique(cluster_labels[cluster_labels >= 0]))
        
        logger.info(f"BiologicalValidator initialized")
        logger.info(f"  Samples: {self.n_samples}")
        logger.info(f"  Genes: {self.n_genes}")
        logger.info(f"  Clusters: {self.n_clusters}")
    
    def external_validation(
        self,
        true_labels: np.ndarray,
        label_names: Optional[Dict[int, str]] = None
    ) -> Dict:
        """
        Compare clusters with known molecular subtypes.
        
        Args:
            true_labels: Known subtype labels (e.g., TRU, PP, PI)
            label_names: Optional mapping of label indices to names
            
        Returns:
            Dictionary with validation metrics
        """
        logger.info("\nPerforming external validation...")
        
        # Compute metrics
        ari = adjusted_rand_score(true_labels, self.cluster_labels)
        nmi = normalized_mutual_info_score(true_labels, self.cluster_labels)
        
        # Compute confusion matrix
        cm = confusion_matrix(true_labels, self.cluster_labels)
        
        results = {
            'adjusted_rand_index': float(ari),
            'normalized_mutual_info': float(nmi),
            'confusion_matrix': cm.tolist(),
            'n_true_labels': len(np.unique(true_labels)),
            'n_predicted_clusters': self.n_clusters
        }
        
        logger.info(f"  Adjusted Rand Index: {ari:.4f}")
        logger.info(f"  Normalized Mutual Info: {nmi:.4f}")
        
        # Interpretation
        if ari >= 0.75:
            interpretation = "Excellent agreement"
        elif ari >= 0.50:
            interpretation = "Good agreement"
        elif ari >= 0.25:
            interpretation = "Moderate agreement"
        else:
            interpretation = "Poor agreement"
        
        results['interpretation'] = interpretation
        logger.info(f"  Interpretation: {interpretation}")
        
        return results
    
    def find_differentially_expressed_genes(
        self,
        cluster_id: int,
        method: str = 'ttest',
        top_n: int = 100,
        fdr_threshold: float = 0.05
    ) -> pd.DataFrame:
        """
        Find genes differentially expressed in a specific cluster.
        
        Args:
            cluster_id: Cluster to analyze
            method: Statistical test ('ttest' or 'mannwhitneyu')
            top_n: Number of top genes to return
            fdr_threshold: FDR threshold for significance
            
        Returns:
            DataFrame with gene names, fold changes, p-values
        """
        logger.info(f"\nFinding DEGs for cluster {cluster_id}...")
        
        # Separate cluster vs rest
        cluster_mask = self.cluster_labels == cluster_id
        cluster_expr = self.gene_expression[cluster_mask]
        rest_expr = self.gene_expression[~cluster_mask]
        
        logger.info(f"  Cluster samples: {cluster_mask.sum()}")
        logger.info(f"  Other samples: {(~cluster_mask).sum()}")
        
        # Compute statistics for each gene
        results = []
        
        for i, gene_name in enumerate(self.gene_names):
            cluster_values = cluster_expr[:, i]
            rest_values = rest_expr[:, i]
            
            # Mean expression
            cluster_mean = cluster_values.mean()
            rest_mean = rest_values.mean()
            
            # Fold change (log2)
            # Add small constant to avoid division by zero
            fold_change = np.log2((cluster_mean + 1e-10) / (rest_mean + 1e-10))
            
            # Statistical test
            if method == 'ttest':
                statistic, pvalue = stats.ttest_ind(cluster_values, rest_values)
            elif method == 'mannwhitneyu':
                statistic, pvalue = stats.mannwhitneyu(cluster_values, rest_values, alternative='two-sided')
            else:
                raise ValueError(f"Unknown method: {method}")
            
            results.append({
                'gene': gene_name,
                'cluster_mean': cluster_mean,
                'rest_mean': rest_mean,
                'log2_fold_change': fold_change,
                'statistic': statistic,
                'pvalue': pvalue
            })
        
        # Create DataFrame
        df = pd.DataFrame(results)
        
        # FDR correction (Benjamini-Hochberg)
        df = df.sort_values('pvalue')
        df['rank'] = range(1, len(df) + 1)
        df['fdr'] = df['pvalue'] * len(df) / df['rank']
        df['fdr'] = df['fdr'].clip(upper=1.0)
        
        # Filter significant genes
        df_sig = df[df['fdr'] < fdr_threshold].copy()
        
        # Sort by absolute fold change
        df_sig['abs_log2fc'] = df_sig['log2_fold_change'].abs()
        df_sig = df_sig.sort_values('abs_log2fc', ascending=False)
        
        # Get top N
        df_top = df_sig.head(top_n)
        
        logger.info(f"  Significant genes (FDR < {fdr_threshold}): {len(df_sig)}")
        logger.info(f"  Returning top {len(df_top)} genes")
        
        if len(df_top) > 0:
            logger.info(f"  Top gene: {df_top.iloc[0]['gene']}, log2FC={df_top.iloc[0]['log2_fold_change']:.2f}")
        else:
            logger.warning("  No significant genes found!")
        
        return df_top[['gene', 'cluster_mean', 'rest_mean', 'log2_fold_change', 'pvalue', 'fdr']]
    
    def find_all_cluster_markers(
        self,
        top_n: int = 50,
        method: str = 'ttest'
    ) -> Dict[int, pd.DataFrame]:
        """
        Find marker genes for all clusters.
        
        Args:
            top_n: Number of top genes per cluster
            method: Statistical test method
            
        Returns:
            Dictionary mapping cluster IDs to marker gene DataFrames
        """
        logger.info("\n" + "="*80)
        logger.info("Finding marker genes for all clusters")
        logger.info("="*80)
        
        all_markers = {}
        
        for cluster_id in range(self.n_clusters):
            markers = self.find_differentially_expressed_genes(
                cluster_id=cluster_id,
                method=method,
                top_n=top_n
            )
            all_markers[cluster_id] = markers
        
        logger.info(f"\n✓ Found markers for {len(all_markers)} clusters")
        
        return all_markers
    
    def get_cluster_gene_signatures(
        self,
        n_genes: int = 10
    ) -> Dict[int, List[str]]:
        """
        Get top marker genes as gene signatures for each cluster.
        
        Args:
            n_genes: Number of genes per signature
            
        Returns:
            Dictionary mapping cluster IDs to gene lists
        """
        logger.info(f"\nGetting gene signatures ({n_genes} genes per cluster)...")
        
        all_markers = self.find_all_cluster_markers(top_n=n_genes)
        
        signatures = {}
        for cluster_id, markers in all_markers.items():
            if len(markers) > 0:
                signatures[cluster_id] = markers['gene'].tolist()[:n_genes]
            else:
                signatures[cluster_id] = []
            
            logger.info(f"  Cluster {cluster_id}: {len(signatures[cluster_id])} genes")
        
        return signatures
    
    def compute_cluster_statistics(self) -> pd.DataFrame:
        """
        Compute basic statistics for each cluster.
        
        Returns:
            DataFrame with cluster statistics
        """
        logger.info("\nComputing cluster statistics...")
        
        stats_list = []
        
        for cluster_id in range(self.n_clusters):
            mask = self.cluster_labels == cluster_id
            cluster_expr = self.gene_expression[mask]
            
            stats_dict = {
                'cluster': cluster_id,
                'n_samples': mask.sum(),
                'percentage': (mask.sum() / self.n_samples) * 100,
                'mean_expression': cluster_expr.mean(),
                'median_expression': np.median(cluster_expr),
                'std_expression': cluster_expr.std(),
                'min_expression': cluster_expr.min(),
                'max_expression': cluster_expr.max()
            }
            
            stats_list.append(stats_dict)
        
        df_stats = pd.DataFrame(stats_list)
        
        logger.info("\nCluster Statistics:")
        for _, row in df_stats.iterrows():
            logger.info(f"  Cluster {int(row['cluster'])}: {int(row['n_samples'])} samples ({row['percentage']:.1f}%)")
        
        return df_stats
    
    def save_results(
        self,
        output_dir: Union[str, Path],
        markers: Optional[Dict[int, pd.DataFrame]] = None,
        external_validation: Optional[Dict] = None
    ):
        """
        Save validation results to disk.
        
        Args:
            output_dir: Directory to save results
            markers: Marker genes per cluster
            external_validation: External validation results
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save cluster statistics
        stats = self.compute_cluster_statistics()
        stats.to_csv(output_dir / 'cluster_statistics.csv', index=False)
        logger.info(f"✓ Saved: {output_dir / 'cluster_statistics.csv'}")
        
        # Save marker genes
        if markers:
            markers_dir = output_dir / 'gene_signatures'
            markers_dir.mkdir(exist_ok=True)
            
            for cluster_id, marker_df in markers.items():
                filename = markers_dir / f'cluster_{cluster_id}_markers.csv'
                marker_df.to_csv(filename, index=False)
            
            logger.info(f"✓ Saved marker genes to: {markers_dir}")
        
        # Save external validation
        if external_validation:
            import json
            with open(output_dir / 'external_validation.json', 'w') as f:
                json.dump(external_validation, f, indent=2)
            logger.info(f"✓ Saved: {output_dir / 'external_validation.json'}")


def compare_with_known_subtypes(
    cluster_labels: np.ndarray,
    known_subtypes: Dict[str, str],
    sample_ids: List[str]
) -> Dict:
    """
    Compare clustering with known LUAD subtypes (TRU, PP, PI).
    
    Args:
        cluster_labels: Predicted cluster assignments
        known_subtypes: Dictionary mapping sample IDs to subtypes
        sample_ids: List of sample IDs in order
        
    Returns:
        Validation results dictionary
    """
    # Map sample IDs to subtype labels
    subtype_labels = []
    valid_indices = []
    
    for i, sample_id in enumerate(sample_ids):
        if sample_id in known_subtypes:
            subtype = known_subtypes[sample_id]
            # Map to numeric labels
            if subtype == 'TRU':
                subtype_labels.append(0)
            elif subtype == 'PP':
                subtype_labels.append(1)
            elif subtype == 'PI':
                subtype_labels.append(2)
            else:
                continue
            valid_indices.append(i)
    
    if len(valid_indices) == 0:
        logger.warning("No samples with known subtypes found!")
        return None
    
    # Filter to valid samples
    true_labels = np.array(subtype_labels)
    pred_labels = cluster_labels[valid_indices]
    
    # Compute metrics
    validator = BiologicalValidator(
        gene_expression=np.zeros((len(valid_indices), 1)),  # Dummy
        cluster_labels=pred_labels
    )
    
    results = validator.external_validation(
        true_labels=true_labels,
        label_names={0: 'TRU', 1: 'PP', 2: 'PI'}
    )
    
    results['n_samples_with_subtypes'] = len(valid_indices)
    
    return results


# Example usage
if __name__ == "__main__":
    print("Biological Validation Module - Phase 8")
    print("="*80)
    
    print("\nExample usage:")
    print("""
    from backend.validation.biological_validation import BiologicalValidator
    import numpy as np
    
    # Load data
    gene_expression = np.load('data/processed/luad_processed.npz')['data']
    cluster_labels = np.load('results/gan_assisted/real_sample_clusters.npz')['kmeans']
    
    # Initialize validator
    validator = BiologicalValidator(gene_expression, cluster_labels)
    
    # Find marker genes for all clusters
    markers = validator.find_all_cluster_markers(top_n=50)
    
    # Get gene signatures
    signatures = validator.get_cluster_gene_signatures(n_genes=10)
    
    # External validation (if known subtypes available)
    # known_labels = load_known_subtypes()
    # validation = validator.external_validation(known_labels)
    
    # Save results
    validator.save_results('results/validation', markers=markers)
    """)
    
    print("\n" + "="*80)
    print("Module ready for use!")