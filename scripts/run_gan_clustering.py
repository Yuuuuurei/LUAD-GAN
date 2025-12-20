"""
GAN-Assisted Clustering Script
Phase 7: Run clustering on GAN-augmented data

This script:
1. Loads augmented data (real + synthetic)
2. Runs multiple clustering algorithms
3. Evaluates clustering quality
4. Compares with baseline results
5. Saves results and visualizations

Usage:
    python scripts/run_gan_clustering.py --data augmented_add_1x.npz
    python scripts/run_gan_clustering.py --data augmented_add_1x.npz --algorithms kmeans hierarchical
    python scripts/run_gan_clustering.py --config configs/clustering_config.yaml

Author: GAN-LUAD Team
Date: 2025
"""

import argparse
import numpy as np
import yaml
import sys
from pathlib import Path
import logging
from datetime import datetime
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backend.clustering.algorithms import ClusteringPipeline, ClusteringComparison
from backend.clustering.evaluation import ClusteringEvaluator, ResultsComparator
from backend.clustering.visualization import (
    ClusterVisualizer, 
    plot_elbow_curve, 
    plot_metrics_comparison,
    plot_improvement_bars
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Run GAN-assisted clustering on augmented LUAD data'
    )
    
    # Data paths
    parser.add_argument(
        '--data-path',
        type=str,
        default='data/synthetic/augmented_add_1x.npz',
        help='Path to augmented data file'
    )
    
    parser.add_argument(
        '--baseline-results',
        type=str,
        default='models/baseline/baseline_clustering_results.json',
        help='Path to baseline clustering results for comparison'
    )
    
    # Clustering parameters
    parser.add_argument(
        '--algorithms',
        type=str,
        nargs='+',
        default=['kmeans', 'hierarchical', 'spectral'],
        choices=['kmeans', 'hierarchical', 'spectral', 'dbscan'],
        help='Clustering algorithms to run'
    )
    
    parser.add_argument(
        '--k-range',
        type=int,
        nargs='+',
        default=[2, 3, 4, 5, 6, 7, 8, 9, 10],
        help='Range of k values to test'
    )
    
    parser.add_argument(
        '--apply-pca',
        action='store_true',
        help='Apply PCA before clustering'
    )
    
    parser.add_argument(
        '--pca-components',
        type=int,
        default=50,
        help='Number of PCA components'
    )
    
    # Output paths
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results/gan_assisted',
        help='Directory to save results'
    )
    
    parser.add_argument(
        '--comparison-dir',
        type=str,
        default='results/comparison',
        help='Directory to save comparison results'
    )
    
    # Visualization
    parser.add_argument(
        '--create-visualizations',
        action='store_true',
        default=True,
        help='Create visualization plots'
    )
    
    # Config file
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to configuration YAML file'
    )
    
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_augmented_data(data_path: str) -> tuple:
    """
    Load augmented data.
    
    Returns:
        data, sample_labels (0=real, 1=synthetic)
    """
    logger.info(f"Loading augmented data from: {data_path}")
    
    data_path = Path(data_path)
    
    if data_path.suffix == '.npz':
        data_dict = np.load(data_path)
        data = data_dict['data']
        sample_labels = data_dict['labels']
    elif data_path.suffix == '.pt':
        import torch
        data_dict = torch.load(data_path)
        data = data_dict['data']
        sample_labels = data_dict['labels']
        if isinstance(data, torch.Tensor):
            data = data.numpy()
        if isinstance(sample_labels, torch.Tensor):
            sample_labels = sample_labels.numpy()
    else:
        raise ValueError(f"Unsupported data format: {data_path.suffix}")
    
    n_real = (sample_labels == 0).sum()
    n_synthetic = (sample_labels == 1).sum()
    
    logger.info(f"Loaded augmented data")
    logger.info(f"  Total samples: {len(data)}")
    logger.info(f"  Real samples: {n_real}")
    logger.info(f"  Synthetic samples: {n_synthetic}")
    logger.info(f"  Features: {data.shape[1]}")
    
    return data, sample_labels


def load_baseline_results(results_path: str) -> dict:
    """Load baseline clustering results for comparison."""
    try:
        with open(results_path, 'r') as f:
            baseline_results = json.load(f)
        logger.info(f"Loaded baseline results from: {results_path}")
        return baseline_results
    except FileNotFoundError:
        logger.warning(f"Baseline results not found at: {results_path}")
        logger.warning("Comparison with baseline will not be available")
        return None


def run_clustering_for_algorithm(
    pipeline: ClusteringPipeline,
    algorithm: str,
    k_range: list,
    output_dir: Path
) -> dict:
    """
    Run clustering for a specific algorithm across multiple k values.
    
    Returns:
        Dictionary of results for each k
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"Running {algorithm.upper()} clustering")
    logger.info(f"{'='*80}")
    
    # Run clustering for all k values
    results = pipeline.run_multiple_k(algorithm=algorithm, k_range=k_range)
    
    # Evaluate each result
    evaluated_results = {}
    
    for k, result in results.items():
        logger.info(f"\nEvaluating {algorithm} with k={k}...")
        
        # Extract cluster labels
        cluster_labels = result['labels']
        
        # Evaluate
        evaluator = ClusteringEvaluator(pipeline.data_scaled, cluster_labels)
        metrics = evaluator.compute_all_metrics()
        
        # Add metrics to result
        result['metrics'] = metrics
        
        # Extract real sample clusters
        real_cluster_labels = pipeline.extract_real_sample_clusters(cluster_labels)
        result['real_sample_labels'] = real_cluster_labels
        
        evaluated_results[k] = result
    
    # Save results
    results_file = output_dir / f'{algorithm}_results.json'
    
    # Prepare for JSON serialization
    serializable_results = {}
    for k, result in evaluated_results.items():
        result_copy = {
            'algorithm': result['algorithm'],
            'n_clusters': result['n_clusters'],
            'metrics': result['metrics'],
            'cluster_sizes': result['cluster_sizes']
        }
        serializable_results[str(k)] = result_copy
    
    with open(results_file, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    logger.info(f"\n✓ {algorithm} results saved to: {results_file}")
    
    return evaluated_results


def create_visualizations(
    data: np.ndarray,
    sample_labels: np.ndarray,
    results: dict,
    algorithm: str,
    optimal_k: int,
    output_dir: Path
):
    """Create visualization plots for clustering results."""
    logger.info(f"\nCreating visualizations for {algorithm} (k={optimal_k})...")
    
    # Get optimal clustering result
    result = results[optimal_k]
    cluster_labels = result['labels']
    
    # Initialize visualizer
    viz = ClusterVisualizer(data, cluster_labels, sample_labels)
    
    # Create output directory
    viz_dir = output_dir / 'visualizations' / algorithm
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    # PCA plot
    viz.plot_pca_clusters(
        save_path=viz_dir / f'pca_k{optimal_k}.png',
        title=f'{algorithm.upper()} Clustering (k={optimal_k}) - PCA',
        show_real_synthetic=True
    )
    
    # t-SNE plot
    viz.plot_tsne_clusters(
        save_path=viz_dir / f'tsne_k{optimal_k}.png',
        title=f'{algorithm.upper()} Clustering (k={optimal_k}) - t-SNE'
    )
    
    # Cluster sizes
    viz.plot_cluster_sizes(
        save_path=viz_dir / f'cluster_sizes_k{optimal_k}.png',
        title=f'{algorithm.upper()} Cluster Sizes (k={optimal_k})'
    )
    
    # Silhouette plot
    if result['metrics']['silhouette_score'] is not None:
        evaluator = ClusteringEvaluator(data, cluster_labels)
        silhouette_vals = evaluator.compute_silhouette_per_sample()
        viz.plot_silhouette(
            silhouette_vals,
            save_path=viz_dir / f'silhouette_k{optimal_k}.png',
            title=f'{algorithm.upper()} Silhouette Analysis (k={optimal_k})'
        )
    
    logger.info(f"✓ Visualizations saved to: {viz_dir}")


def compare_with_baseline(
    gan_results: dict,
    baseline_results: dict,
    algorithm: str,
    output_dir: Path
) -> dict:
    """
    Compare GAN-assisted results with baseline.
    
    Returns:
        Dictionary of improvements
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"Comparing {algorithm} with baseline")
    logger.info(f"{'='*80}")
    
    comparator = ResultsComparator()
    
    # Add results for each k
    for k_str, gan_result in gan_results.items():
        k = int(k_str)
        
        # Check if baseline has this k
        baseline_key = f"{algorithm}_k{k}"
        if baseline_key in baseline_results:
            comparator.add_baseline_result(
                f'k{k}',
                baseline_results[baseline_key]['metrics']
            )
            comparator.add_gan_result(
                f'k{k}',
                gan_result['metrics']
            )
    
    # Compute improvements
    improvements = comparator.compute_improvements()
    
    # Print summary
    comparator.print_summary()
    
    # Save comparison table
    comparison_file = output_dir / f'{algorithm}_comparison.csv'
    comparator.save_comparison(comparison_file)
    
    # Save improvements JSON
    improvements_file = output_dir / f'{algorithm}_improvements.json'
    with open(improvements_file, 'w') as f:
        json.dump(improvements, f, indent=2)
    
    logger.info(f"\n✓ Comparison saved to: {comparison_file}")
    
    return improvements


def main():
    """Main execution function."""
    args = parse_args()
    
    # Load config if provided
    if args.config:
        logger.info(f"Loading configuration from: {args.config}")
        config = load_config(args.config)
        for key, value in config.items():
            if hasattr(args, key):
                setattr(args, key, value)
    
    # Print configuration
    logger.info("\n" + "="*80)
    logger.info("GAN-Assisted Clustering - Phase 7")
    logger.info("="*80)
    logger.info(f"Data: {args.data_path}")
    logger.info(f"Algorithms: {args.algorithms}")
    logger.info(f"K range: {args.k_range}")
    logger.info(f"Apply PCA: {args.apply_pca}")
    
    # Create output directories
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    comparison_dir = Path(args.comparison_dir)
    comparison_dir.mkdir(parents=True, exist_ok=True)
    
    # Load augmented data
    data, sample_labels = load_augmented_data(args.data_path)
    
    # Load baseline results
    baseline_results = load_baseline_results(args.baseline_results)
    
    # Initialize clustering pipeline
    pipeline = ClusteringPipeline(data, sample_labels=sample_labels, scale_data=True)
    
    # Apply PCA if requested
    if args.apply_pca:
        logger.info(f"\nApplying PCA ({args.pca_components} components)...")
        pipeline.apply_pca_before_clustering(n_components=args.pca_components)
    
    # Run clustering for each algorithm
    all_results = {}
    all_improvements = {}
    
    for algorithm in args.algorithms:
        # Run clustering
        results = run_clustering_for_algorithm(
            pipeline=pipeline,
            algorithm=algorithm,
            k_range=args.k_range,
            output_dir=output_dir
        )
        
        all_results[algorithm] = results
        
        # Find optimal k (based on silhouette score)
        optimal_k = max(
            results.keys(),
            key=lambda k: results[k]['metrics'].get('silhouette_score', -1)
        )
        
        logger.info(f"\n✓ Optimal k for {algorithm}: {optimal_k}")
        logger.info(f"  Silhouette Score: {results[optimal_k]['metrics']['silhouette_score']:.4f}")
        
        # Create visualizations
        if args.create_visualizations:
            create_visualizations(
                data=pipeline.data_scaled,
                sample_labels=sample_labels,
                results=results,
                algorithm=algorithm,
                optimal_k=optimal_k,
                output_dir=output_dir
            )
        
        # Compare with baseline
        if baseline_results:
            improvements = compare_with_baseline(
                gan_results=results,
                baseline_results=baseline_results,
                algorithm=algorithm,
                output_dir=comparison_dir
            )
            all_improvements[algorithm] = improvements
            
            # Plot improvements
            if improvements and args.create_visualizations:
                # Get improvements for optimal k
                optimal_improvements = improvements.get(f'k{optimal_k}', {})
                if optimal_improvements:
                    plot_improvement_bars(
                        optimal_improvements,
                        save_path=comparison_dir / f'{algorithm}_improvements_k{optimal_k}.png',
                        title=f'{algorithm.upper()} Improvements (k={optimal_k})'
                    )
    
    # Save summary
    summary = {
        'data_path': args.data_path,
        'algorithms': args.algorithms,
        'k_range': args.k_range,
        'apply_pca': args.apply_pca,
        'timestamp': datetime.now().isoformat(),
        'results_summary': {}
    }
    
    for algorithm, results in all_results.items():
        optimal_k = max(
            results.keys(),
            key=lambda k: results[k]['metrics'].get('silhouette_score', -1)
        )
        
        summary['results_summary'][algorithm] = {
            'optimal_k': optimal_k,
            'optimal_metrics': results[optimal_k]['metrics']
        }
    
    summary_file = output_dir / 'clustering_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info("\n" + "="*80)
    logger.info("✓ GAN-Assisted Clustering Complete!")
    logger.info("="*80)
    logger.info(f"Results saved to: {output_dir}")
    if baseline_results:
        logger.info(f"Comparisons saved to: {comparison_dir}")
    logger.info("="*80)


if __name__ == "__main__":
    main()