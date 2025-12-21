"""
Survival Analysis Module for GAN-LUAD Clustering Project
Phase 8: Analyze survival differences between clusters

This module provides:
- Kaplan-Meier survival curves
- Log-rank test for survival differences
- Cox proportional hazards model
- Survival visualization

Requires: lifelines library
Install: pip install lifelines

Author: GAN-LUAD Team
Date: 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import logging
import warnings
warnings.filterwarnings('ignore')

# Try to import lifelines (optional dependency)
try:
    from lifelines import KaplanMeierFitter, CoxPHFitter
    from lifelines.statistics import logrank_test, multivariate_logrank_test
    LIFELINES_AVAILABLE = True
except ImportError:
    LIFELINES_AVAILABLE = False
    logging.warning("lifelines not installed. Survival analysis will not be available.")
    logging.warning("Install with: pip install lifelines")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SurvivalAnalyzer:
    """
    Analyze survival differences between clusters.
    """
    
    def __init__(
        self,
        cluster_labels: np.ndarray,
        survival_time: np.ndarray,
        vital_status: np.ndarray,
        sample_ids: Optional[List[str]] = None
    ):
        """
        Initialize survival analyzer.
        
        Args:
            cluster_labels: Cluster assignments
            survival_time: Survival time in days (or months)
            vital_status: Event indicator (1=death, 0=censored)
            sample_ids: Optional sample IDs
        """
        if not LIFELINES_AVAILABLE:
            raise ImportError("lifelines is required for survival analysis. Install with: pip install lifelines")
        
        self.cluster_labels = cluster_labels
        self.survival_time = survival_time
        self.vital_status = vital_status
        self.sample_ids = sample_ids if sample_ids else [f"Sample_{i}" for i in range(len(cluster_labels))]
        
        self.n_samples = len(cluster_labels)
        self.n_clusters = len(np.unique(cluster_labels[cluster_labels >= 0]))
        
        # Create DataFrame for analysis
        self.survival_df = pd.DataFrame({
            'cluster': cluster_labels,
            'time': survival_time,
            'event': vital_status,
            'sample_id': self.sample_ids
        })
        
        # Remove any invalid entries
        self.survival_df = self.survival_df[
            (self.survival_df['time'] > 0) & 
            (self.survival_df['cluster'] >= 0)
        ]
        
        logger.info(f"SurvivalAnalyzer initialized")
        logger.info(f"  Samples: {len(self.survival_df)}")
        logger.info(f"  Events (deaths): {self.survival_df['event'].sum()}")
        logger.info(f"  Censored: {(1 - self.survival_df['event']).sum()}")
        logger.info(f"  Clusters: {self.n_clusters}")
    
    def compute_kaplan_meier(self) -> Dict[int, KaplanMeierFitter]:
        """
        Compute Kaplan-Meier survival curves for each cluster.
        
        Returns:
            Dictionary mapping cluster IDs to fitted KM models
        """
        logger.info("\nComputing Kaplan-Meier curves...")
        
        km_models = {}
        
        for cluster_id in range(self.n_clusters):
            cluster_df = self.survival_df[self.survival_df['cluster'] == cluster_id]
            
            if len(cluster_df) == 0:
                logger.warning(f"  Cluster {cluster_id}: No samples!")
                continue
            
            kmf = KaplanMeierFitter()
            kmf.fit(
                durations=cluster_df['time'],
                event_observed=cluster_df['event'],
                label=f'Cluster {cluster_id}'
            )
            
            km_models[cluster_id] = kmf
            
            # Log statistics
            median_survival = kmf.median_survival_time_
            logger.info(f"  Cluster {cluster_id}: n={len(cluster_df)}, "
                       f"events={cluster_df['event'].sum()}, "
                       f"median survival={median_survival:.1f}")
        
        return km_models
    
    def logrank_test(self) -> Dict:
        """
        Perform log-rank test to compare survival across clusters.
        
        Returns:
            Dictionary with test statistics and p-value
        """
        logger.info("\nPerforming log-rank test...")
        
        if self.n_clusters < 2:
            logger.warning("Need at least 2 clusters for log-rank test")
            return None
        
        # Perform multivariate log-rank test
        results = multivariate_logrank_test(
            durations=self.survival_df['time'],
            groups=self.survival_df['cluster'],
            event_observed=self.survival_df['event']
        )
        
        test_results = {
            'test_statistic': float(results.test_statistic),
            'p_value': float(results.p_value),
            'degrees_of_freedom': int(results.degrees_of_freedom)
        }
        
        logger.info(f"  Test statistic: {results.test_statistic:.4f}")
        logger.info(f"  P-value: {results.p_value:.4f}")
        
        # Interpretation
        if results.p_value < 0.001:
            interpretation = "Highly significant differences"
        elif results.p_value < 0.01:
            interpretation = "Significant differences"
        elif results.p_value < 0.05:
            interpretation = "Marginally significant differences"
        else:
            interpretation = "No significant differences"
        
        test_results['interpretation'] = interpretation
        logger.info(f"  Interpretation: {interpretation}")
        
        return test_results
    
    def pairwise_logrank(self) -> pd.DataFrame:
        """
        Perform pairwise log-rank tests between all cluster pairs.
        
        Returns:
            DataFrame with pairwise comparisons
        """
        logger.info("\nPerforming pairwise log-rank tests...")
        
        comparisons = []
        
        for i in range(self.n_clusters):
            for j in range(i + 1, self.n_clusters):
                cluster_i = self.survival_df[self.survival_df['cluster'] == i]
                cluster_j = self.survival_df[self.survival_df['cluster'] == j]
                
                if len(cluster_i) == 0 or len(cluster_j) == 0:
                    continue
                
                # Combine data
                combined_df = pd.concat([cluster_i, cluster_j])
                
                # Perform test
                results = multivariate_logrank_test(
                    durations=combined_df['time'],
                    groups=combined_df['cluster'],
                    event_observed=combined_df['event']
                )
                
                comparisons.append({
                    'cluster_1': i,
                    'cluster_2': j,
                    'n_1': len(cluster_i),
                    'n_2': len(cluster_j),
                    'test_statistic': results.test_statistic,
                    'p_value': results.p_value,
                    'significant': results.p_value < 0.05
                })
                
                logger.info(f"  Cluster {i} vs {j}: p={results.p_value:.4f}")
        
        return pd.DataFrame(comparisons)
    
    def cox_proportional_hazards(self) -> Dict:
        """
        Fit Cox proportional hazards model with cluster as covariate.
        
        Returns:
            Dictionary with model results
        """
        logger.info("\nFitting Cox proportional hazards model...")
        
        # Create dummy variables for clusters
        cluster_dummies = pd.get_dummies(self.survival_df['cluster'], prefix='cluster')
        
        # Combine with survival data
        cox_df = pd.concat([
            self.survival_df[['time', 'event']],
            cluster_dummies
        ], axis=1)
        
        # Fit model
        cph = CoxPHFitter()
        cph.fit(cox_df, duration_col='time', event_col='event')
        
        # Extract results
        results = {
            'concordance_index': float(cph.concordance_index_),
            'log_likelihood': float(cph.log_likelihood_),
            'AIC': float(cph.AIC_),
            'coefficients': cph.params_.to_dict(),
            'hazard_ratios': np.exp(cph.params_).to_dict(),
            'p_values': cph.summary['p'].to_dict(),
            'significant_covariates': []
        }
        
        # Find significant covariates
        for covariate, p_value in cph.summary['p'].items():
            if p_value < 0.05:
                results['significant_covariates'].append(covariate)
        
        logger.info(f"  Concordance index: {cph.concordance_index_:.4f}")
        logger.info(f"  Significant covariates: {len(results['significant_covariates'])}")
        
        return results
    
    def plot_kaplan_meier(
        self,
        save_path: Optional[Union[str, Path]] = None,
        title: str = "Kaplan-Meier Survival Curves by Cluster",
        figsize: Tuple[int, int] = (12, 8)
    ):
        """
        Plot Kaplan-Meier survival curves for all clusters.
        
        Args:
            save_path: Optional path to save figure
            title: Plot title
            figsize: Figure size
        """
        logger.info("\nPlotting Kaplan-Meier curves...")
        
        # Compute KM curves
        km_models = self.compute_kaplan_meier()
        
        if len(km_models) == 0:
            logger.warning("No KM models to plot!")
            return
        
        # Create plot
        fig, ax = plt.subplots(figsize=figsize)
        
        colors = sns.color_palette("husl", self.n_clusters)
        
        for cluster_id, kmf in km_models.items():
            kmf.plot_survival_function(
                ax=ax,
                ci_show=True,
                color=colors[cluster_id],
                linewidth=2.5,
                alpha=0.8
            )
        
        # Perform log-rank test and add to plot
        logrank_results = self.logrank_test()
        if logrank_results:
            p_value = logrank_results['p_value']
            ax.text(
                0.02, 0.02,
                f"Log-rank test: p = {p_value:.4f}",
                transform=ax.transAxes,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                fontsize=11,
                verticalalignment='bottom'
            )
        
        ax.set_xlabel('Time (days)', fontsize=12)
        ax.set_ylabel('Survival Probability', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"✓ Kaplan-Meier plot saved to: {save_path}")
        
        plt.show()
    
    def plot_cumulative_density(
        self,
        save_path: Optional[Union[str, Path]] = None,
        title: str = "Cumulative Event Density by Cluster",
        figsize: Tuple[int, int] = (12, 8)
    ):
        """
        Plot cumulative event density (complement of survival).
        
        Args:
            save_path: Optional path to save figure
            title: Plot title
            figsize: Figure size
        """
        logger.info("\nPlotting cumulative density...")
        
        km_models = self.compute_kaplan_meier()
        
        fig, ax = plt.subplots(figsize=figsize)
        colors = sns.color_palette("husl", self.n_clusters)
        
        for cluster_id, kmf in km_models.items():
            kmf.plot_cumulative_density(
                ax=ax,
                ci_show=True,
                color=colors[cluster_id],
                linewidth=2.5,
                alpha=0.8
            )
        
        ax.set_xlabel('Time (days)', fontsize=12)
        ax.set_ylabel('Cumulative Event Probability', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"✓ Cumulative density plot saved to: {save_path}")
        
        plt.show()
    
    def generate_summary_table(self) -> pd.DataFrame:
        """
        Generate summary table of survival statistics per cluster.
        
        Returns:
            DataFrame with survival statistics
        """
        logger.info("\nGenerating survival summary table...")
        
        km_models = self.compute_kaplan_meier()
        
        summary_data = []
        
        for cluster_id in range(self.n_clusters):
            cluster_df = self.survival_df[self.survival_df['cluster'] == cluster_id]
            
            if len(cluster_df) == 0:
                continue
            
            kmf = km_models.get(cluster_id)
            
            summary = {
                'cluster': cluster_id,
                'n_samples': len(cluster_df),
                'n_events': cluster_df['event'].sum(),
                'n_censored': (1 - cluster_df['event']).sum(),
                'median_survival': kmf.median_survival_time_ if kmf else np.nan,
                'mean_survival_time': cluster_df['time'].mean(),
                'event_rate': (cluster_df['event'].sum() / len(cluster_df)) * 100
            }
            
            summary_data.append(summary)
        
        df = pd.DataFrame(summary_data)
        
        logger.info("\nSurvival Summary:")
        for _, row in df.iterrows():
            logger.info(f"  Cluster {int(row['cluster'])}: "
                       f"n={int(row['n_samples'])}, "
                       f"events={int(row['n_events'])}, "
                       f"median={row['median_survival']:.1f}")
        
        return df
    
    def save_results(
        self,
        output_dir: Union[str, Path],
        save_plots: bool = True
    ):
        """
        Save all survival analysis results.
        
        Args:
            output_dir: Directory to save results
            save_plots: Whether to save plots
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save summary table
        summary = self.generate_summary_table()
        summary.to_csv(output_dir / 'survival_summary.csv', index=False)
        logger.info(f"✓ Saved: {output_dir / 'survival_summary.csv'}")
        
        # Save log-rank test results
        logrank = self.logrank_test()
        if logrank:
            import json
            with open(output_dir / 'logrank_test.json', 'w') as f:
                json.dump(logrank, f, indent=2)
            logger.info(f"✓ Saved: {output_dir / 'logrank_test.json'}")
        
        # Save pairwise comparisons
        pairwise = self.pairwise_logrank()
        if len(pairwise) > 0:
            pairwise.to_csv(output_dir / 'pairwise_logrank.csv', index=False)
            logger.info(f"✓ Saved: {output_dir / 'pairwise_logrank.csv'}")
        
        # Save Cox model results
        try:
            cox_results = self.cox_proportional_hazards()
            import json
            with open(output_dir / 'cox_model_results.json', 'w') as f:
                json.dump(cox_results, f, indent=2)
            logger.info(f"✓ Saved: {output_dir / 'cox_model_results.json'}")
        except Exception as e:
            logger.warning(f"Could not fit Cox model: {e}")
        
        # Save plots
        if save_plots:
            self.plot_kaplan_meier(save_path=output_dir / 'kaplan_meier_curves.png')
            self.plot_cumulative_density(save_path=output_dir / 'cumulative_density.png')


# Example usage
if __name__ == "__main__":
    print("Survival Analysis Module - Phase 8")
    print("="*80)
    
    if not LIFELINES_AVAILABLE:
        print("\n⚠️  lifelines not installed!")
        print("Install with: pip install lifelines")
    else:
        print("\n✓ lifelines available")
    
    print("\nExample usage:")
    print("""
    from backend.validation.survival_analysis import SurvivalAnalyzer
    import numpy as np
    import pandas as pd
    
    # Load clinical data
    clinical = pd.read_csv('data/raw/clinical_data.tsv', sep='\\t')
    
    # Extract survival information
    survival_time = clinical['OS.time'].values  # Overall survival time
    vital_status = clinical['OS'].values  # 1=death, 0=censored
    
    # Load cluster labels
    cluster_labels = np.load('results/gan_assisted/real_sample_clusters.npz')['kmeans']
    
    # Initialize analyzer
    analyzer = SurvivalAnalyzer(cluster_labels, survival_time, vital_status)
    
    # Perform analyses
    km_curves = analyzer.compute_kaplan_meier()
    logrank = analyzer.logrank_test()
    cox_results = analyzer.cox_proportional_hazards()
    
    # Create visualizations
    analyzer.plot_kaplan_meier(save_path='results/validation/kaplan_meier.png')
    
    # Save all results
    analyzer.save_results('results/validation/survival_analysis')
    """)
    
    print("\n" + "="*80)
    print("Module ready for use!")