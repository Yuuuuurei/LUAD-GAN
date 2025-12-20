"""
Results Evaluation Script
Phase 7: Evaluate and compare clustering results

This script:
1. Loads baseline and GAN-assisted clustering results
2. Computes improvement metrics
3. Generates comparison visualizations
4. Creates comprehensive evaluation report

Usage:
    python scripts/evaluate_results.py
    python scripts/evaluate_results.py --baseline-dir models/baseline --gan-dir results/gan_assisted

Author: GAN-LUAD Team
Date: 2025
"""

import argparse
import numpy as np
import json
import sys
from pathlib import Path
import logging
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backend.clustering.evaluation import ResultsComparator
from backend.clustering.visualization import plot_metrics_comparison, plot_improvement_bars

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Evaluate and compare clustering results'
    )
    
    parser.add_argument(
        '--baseline-dir',
        type=str,
        default='models/baseline',
        help='Directory containing baseline results'
    )
    
    parser.add_argument(
        '--gan-dir',
        type=str,
        default='results/gan_assisted',
        help='Directory containing GAN-assisted results'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results/comparison',
        help='Directory to save evaluation results'
    )
    
    parser.add_argument(
        '--algorithms',
        type=str,
        nargs='+',
        default=['kmeans', 'hierarchical', 'spectral'],
        help='Algorithms to evaluate'
    )
    
    return parser.parse_args()


def load_results_file(file_path: Path) -> dict:
    """Load clustering results from JSON file."""
    try:
        with open(file_path, 'r') as f:
            results = json.load(f)
        logger.info(f"Loaded: {file_path}")
        return results
    except FileNotFoundError:
        logger.warning(f"File not found: {file_path}")
        return None


def create_comprehensive_comparison(
    baseline_dir: Path,
    gan_dir: Path,
    algorithms: list
) -> pd.DataFrame:
    """
    Create comprehensive comparison DataFrame.
    
    Returns:
        DataFrame with all metrics for baseline and GAN-assisted
    """
    logger.info("\nCreating comprehensive comparison...")
    
    rows = []
    
    for algorithm in algorithms:
        # Load results
        baseline_file = baseline_dir / f'{algorithm}_results.json'
        gan_file = gan_dir / f'{algorithm}_results.json'
        
        baseline_results = load_results_file(baseline_file)
        gan_results = load_results_file(gan_file)
        
        if baseline_results is None or gan_results is None:
            logger.warning(f"Skipping {algorithm} - missing results")
            continue
        
        # Compare each k
        for k_str in baseline_results.keys():
            if k_str not in gan_results:
                continue
            
            k = int(k_str)
            baseline_metrics = baseline_results[k_str].get('metrics', {})
            gan_metrics = gan_results[k_str].get('metrics', {})
            
            row = {
                'algorithm': algorithm,
                'k': k,
                'baseline_silhouette': baseline_metrics.get('silhouette_score'),
                'gan_silhouette': gan_metrics.get('silhouette_score'),
                'baseline_davies_bouldin': baseline_metrics.get('davies_bouldin_index'),
                'gan_davies_bouldin': gan_metrics.get('davies_bouldin_index'),
                'baseline_calinski_harabasz': baseline_metrics.get('calinski_harabasz_score'),
                'gan_calinski_harabasz': gan_metrics.get('calinski_harabasz_score'),
            }
            
            # Compute improvements
            if row['baseline_silhouette'] and row['gan_silhouette']:
                row['silhouette_improvement_%'] = (
                    (row['gan_silhouette'] - row['baseline_silhouette']) / 
                    abs(row['baseline_silhouette']) * 100
                )
            
            if row['baseline_davies_bouldin'] and row['gan_davies_bouldin']:
                row['davies_bouldin_improvement_%'] = (
                    (row['baseline_davies_bouldin'] - row['gan_davies_bouldin']) / 
                    abs(row['baseline_davies_bouldin']) * 100
                )
            
            if row['baseline_calinski_harabasz'] and row['gan_calinski_harabasz']:
                row['calinski_harabasz_improvement_%'] = (
                    (row['gan_calinski_harabasz'] - row['baseline_calinski_harabasz']) / 
                    abs(row['baseline_calinski_harabasz']) * 100
                )
            
            rows.append(row)
    
    df = pd.DataFrame(rows)
    logger.info(f"✓ Created comparison with {len(df)} configurations")
    
    return df


def plot_comprehensive_comparison(df: pd.DataFrame, output_dir: Path):
    """Create comprehensive comparison visualizations."""
    logger.info("\nCreating comprehensive visualizations...")
    
    # 1. Heatmap of improvements
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    metrics = ['silhouette', 'davies_bouldin', 'calinski_harabasz']
    metric_names = ['Silhouette', 'Davies-Bouldin', 'Calinski-Harabasz']
    
    for ax, metric, name in zip(axes, metrics, metric_names):
        # Pivot table for heatmap
        pivot = df.pivot_table(
            values=f'{metric}_improvement_%',
            index='algorithm',
            columns='k',
            aggfunc='mean'
        )
        
        sns.heatmap(
            pivot,
            annot=True,
            fmt='.2f',
            cmap='RdYlGn',
            center=0,
            ax=ax,
            cbar_kws={'label': 'Improvement (%)'}
        )
        
        ax.set_title(f'{name} Improvement (%)')
        ax.set_xlabel('Number of Clusters (k)')
        ax.set_ylabel('Algorithm')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'improvement_heatmaps.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info("✓ Heatmap saved")
    
    # 2. Line plots showing improvement across k
    for algorithm in df['algorithm'].unique():
        algo_df = df[df['algorithm'] == algorithm]
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        for ax, metric, name in zip(axes, metrics, metric_names):
            improvement_col = f'{metric}_improvement_%'
            
            if improvement_col in algo_df.columns:
                ax.plot(algo_df['k'], algo_df[improvement_col], 'o-', linewidth=2, markersize=8)
                ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
                ax.set_xlabel('Number of Clusters (k)')
                ax.set_ylabel('Improvement (%)')
                ax.set_title(f'{name}')
                ax.grid(True, alpha=0.3)
        
        fig.suptitle(f'{algorithm.upper()} - Improvements Across k', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_dir / f'{algorithm}_improvement_trends.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    logger.info("✓ Trend plots saved")
    
    # 3. Best configurations summary
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Find best k for each algorithm (by silhouette improvement)
    best_configs = []
    for algorithm in df['algorithm'].unique():
        algo_df = df[df['algorithm'] == algorithm]
        if 'silhouette_improvement_%' in algo_df.columns:
            best_idx = algo_df['silhouette_improvement_%'].idxmax()
            best_row = algo_df.loc[best_idx]
            best_configs.append({
                'algorithm': algorithm,
                'k': int(best_row['k']),
                'improvement': best_row['silhouette_improvement_%']
            })
    
    if best_configs:
        best_df = pd.DataFrame(best_configs)
        bars = ax.bar(
            best_df['algorithm'],
            best_df['improvement'],
            color='steelblue',
            alpha=0.7,
            edgecolor='black'
        )
        
        # Add value labels and k labels
        for bar, k, imp in zip(bars, best_df['k'], best_df['improvement']):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.,
                height,
                f'{imp:.2f}%\n(k={k})',
                ha='center',
                va='bottom',
                fontweight='bold'
            )
        
        ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        ax.set_xlabel('Algorithm')
        ax.set_ylabel('Best Silhouette Improvement (%)')
        ax.set_title('Best Clustering Configuration per Algorithm')
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'best_configurations.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info("✓ Best configurations plot saved")


def generate_evaluation_report(
    df: pd.DataFrame,
    output_dir: Path
):
    """Generate comprehensive evaluation report."""
    logger.info("\nGenerating evaluation report...")
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'summary': {},
        'detailed_results': []
    }
    
    # Overall statistics
    report['summary']['total_configurations'] = len(df)
    report['summary']['algorithms_evaluated'] = df['algorithm'].unique().tolist()
    report['summary']['k_values_tested'] = sorted(df['k'].unique().tolist())
    
    # Average improvements
    for metric in ['silhouette', 'davies_bouldin', 'calinski_harabasz']:
        col = f'{metric}_improvement_%'
        if col in df.columns:
            mean_imp = df[col].mean()
            report['summary'][f'mean_{metric}_improvement_%'] = float(mean_imp)
    
    # Best configurations
    best_configs = []
    for algorithm in df['algorithm'].unique():
        algo_df = df[df['algorithm'] == algorithm]
        
        if 'silhouette_improvement_%' in algo_df.columns:
            best_idx = algo_df['silhouette_improvement_%'].idxmax()
            best_row = algo_df.loc[best_idx]
            
            best_configs.append({
                'algorithm': algorithm,
                'k': int(best_row['k']),
                'silhouette_improvement': float(best_row['silhouette_improvement_%']),
                'baseline_silhouette': float(best_row['baseline_silhouette']),
                'gan_silhouette': float(best_row['gan_silhouette'])
            })
    
    report['best_configurations'] = best_configs
    
    # Detailed results
    for _, row in df.iterrows():
        result = {
            'algorithm': row['algorithm'],
            'k': int(row['k']),
            'metrics': {}
        }
        
        for metric in ['silhouette', 'davies_bouldin', 'calinski_harabasz']:
            result['metrics'][metric] = {
                'baseline': float(row[f'baseline_{metric}']) if pd.notna(row[f'baseline_{metric}']) else None,
                'gan': float(row[f'gan_{metric}']) if pd.notna(row[f'gan_{metric}']) else None,
                'improvement_%': float(row[f'{metric}_improvement_%']) if pd.notna(row[f'{metric}_improvement_%']) else None
            }
        
        report['detailed_results'].append(result)
    
    # Save report
    report_file = output_dir / 'evaluation_report.json'
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"✓ Evaluation report saved to: {report_file}")
    
    # Print summary
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)
    print(f"\nTotal configurations evaluated: {report['summary']['total_configurations']}")
    print(f"Algorithms: {', '.join(report['summary']['algorithms_evaluated'])}")
    print(f"K values tested: {report['summary']['k_values_tested']}")
    
    print("\nAverage Improvements:")
    for metric in ['silhouette', 'davies_bouldin', 'calinski_harabasz']:
        key = f'mean_{metric}_improvement_%'
        if key in report['summary']:
            print(f"  {metric}: {report['summary'][key]:+.2f}%")
    
    print("\nBest Configurations:")
    for config in best_configs:
        print(f"\n  {config['algorithm'].upper()} (k={config['k']}):")
        print(f"    Baseline Silhouette: {config['baseline_silhouette']:.4f}")
        print(f"    GAN Silhouette: {config['gan_silhouette']:.4f}")
        print(f"    Improvement: {config['silhouette_improvement']:+.2f}%")
    
    print("\n" + "="*80)


def main():
    """Main execution function."""
    args = parse_args()
    
    logger.info("\n" + "="*80)
    logger.info("Results Evaluation - Phase 7")
    logger.info("="*80)
    logger.info(f"Baseline directory: {args.baseline_dir}")
    logger.info(f"GAN directory: {args.gan_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create comprehensive comparison
    baseline_dir = Path(args.baseline_dir)
    gan_dir = Path(args.gan_dir)
    
    df = create_comprehensive_comparison(
        baseline_dir=baseline_dir,
        gan_dir=gan_dir,
        algorithms=args.algorithms
    )
    
    # Save comparison table
    comparison_file = output_dir / 'metrics_comparison.csv'
    df.to_csv(comparison_file, index=False)
    logger.info(f"\n✓ Comparison table saved to: {comparison_file}")
    
    # Create visualizations
    plot_comprehensive_comparison(df, output_dir)
    
    # Generate evaluation report
    generate_evaluation_report(df, output_dir)
    
    logger.info("\n" + "="*80)
    logger.info("✓ Evaluation Complete!")
    logger.info("="*80)
    logger.info(f"All results saved to: {output_dir}")
    logger.info("="*80)


if __name__ == "__main__":
    main()