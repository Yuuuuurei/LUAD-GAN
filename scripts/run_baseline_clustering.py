"""
Script to run baseline clustering experiments.
Tests multiple algorithms and saves results.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import argparse
import yaml
import torch
import numpy as np
import pandas as pd
from datetime import datetime

from backend.clustering import ClusteringPipeline, ClusteringEvaluator, ClusteringVisualizer
from backend.config import PROCESSED_DATA_DIR, RANDOM_SEED
from backend.utils import setup_logging, save_json, set_random_seeds


def load_config(config_path: Path) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def run_baseline_clustering(config: dict, logger):
    """
    Main baseline clustering function.
    
    Args:
        config: Configuration dictionary
        logger: Logger instance
    """
    logger.info("=" * 70)
    logger.info("Baseline Clustering Experiments")
    logger.info("=" * 70)
    
    # Set random seeds
    set_random_seeds(config['reproducibility']['random_seed'])
    
    # ========================================================================
    # Step 1: Load Processed Data
    # ========================================================================
    logger.info("\n[1/6] Loading processed data...")
    
    data_file = Path(config['input']['processed_data_file'])
    data_tensor = torch.load(data_file)
    data = data_tensor.numpy()
    
    logger.info(f"Loaded data shape: {data.shape}")
    logger.info(f"  Samples: {data.shape[0]}")
    logger.info(f"  Features: {data.shape[1]}")
    
    # Load sample IDs and feature names
    sample_ids_file = Path(config['input']['sample_ids_file'])
    with open(sample_ids_file, 'r') as f:
        sample_ids = [line.strip() for line in f]
    
    feature_names_file = Path(config['input']['feature_names_file'])
    with open(feature_names_file, 'r') as f:
        feature_names = [line.strip() for line in f]
    
    logger.info(f"Loaded {len(sample_ids)} sample IDs and {len(feature_names)} feature names")
    
    # ========================================================================
    # Step 2: Initialize Components
    # ========================================================================
    logger.info("\n[2/6] Initializing clustering components...")
    
    pipeline = ClusteringPipeline(
        random_state=config['reproducibility']['random_seed'],
        n_jobs=config['performance']['n_jobs'],
        logger=logger
    )
    
    evaluator = ClusteringEvaluator(logger=logger)
    
    viz_dir = Path(config['output']['visualizations_dir'])
    visualizer = ClusteringVisualizer(
        output_dir=viz_dir,
        dpi=config['visualization']['dpi'],
        logger=logger
    )
    
    # ========================================================================
    # Step 3: Dimensionality Reduction (Optional)
    # ========================================================================
    logger.info("\n[3/6] Applying dimensionality reduction...")
    
    # PCA for clustering
    if config['dimensionality_reduction']['pca']['enabled']:
        n_components = config['dimensionality_reduction']['pca']['n_components']
        pca_data, pca_model = pipeline.apply_pca(data, n_components=n_components)
        clustering_data = pca_data
        logger.info(f"Using PCA-reduced data for clustering: {clustering_data.shape}")
    else:
        clustering_data = data
        logger.info("Using original data for clustering")
    
    # t-SNE for visualization
    if config['dimensionality_reduction']['tsne']['enabled']:
        tsne_embedding = pipeline.apply_tsne(
            clustering_data,
            n_components=config['dimensionality_reduction']['tsne']['n_components'],
            perplexity=config['dimensionality_reduction']['tsne']['perplexity']
        )
        logger.info(f"t-SNE embedding created: {tsne_embedding.shape}")
    else:
        tsne_embedding = None
    
    # UMAP for visualization
    if config['dimensionality_reduction']['umap']['enabled']:
        umap_embedding = pipeline.apply_umap(
            clustering_data,
            n_components=config['dimensionality_reduction']['umap']['n_components'],
            n_neighbors=config['dimensionality_reduction']['umap']['n_neighbors']
        )
        if umap_embedding is not None:
            logger.info(f"UMAP embedding created: {umap_embedding.shape}")
    else:
        umap_embedding = None
    
    # ========================================================================
    # Step 4: Run Clustering Algorithms
    # ========================================================================
    logger.info("\n[4/6] Running clustering algorithms...")
    
    all_results = {}
    k_range = config['clustering']['kmeans']['n_clusters_range']
    
    # K-Means Clustering
    if config['clustering']['kmeans']['enabled']:
        logger.info("\n--- K-Means Clustering ---")
        kmeans_results = {}
        
        for k in k_range:
            labels, model = pipeline.kmeans_clustering(
                clustering_data,
                n_clusters=k,
                n_init=config['clustering']['kmeans']['n_init'],
                max_iter=config['clustering']['kmeans']['max_iter']
            )
            
            # Evaluate
            metrics = evaluator.compute_all_metrics(clustering_data, labels)
            metrics['k'] = k
            metrics['inertia'] = model.inertia_
            
            model_name = f'kmeans_k{k}'
            kmeans_results[k] = (labels, metrics)
            all_results[model_name] = metrics
            
            logger.info(f"  k={k}: Silhouette={metrics['silhouette_score']:.4f}, "
                       f"DB={metrics['davies_bouldin_index']:.4f}")
        
        # Save K-Means results
        results_dir = Path(config['output']['results_dir'])
        results_dir.mkdir(parents=True, exist_ok=True)
        
        kmeans_file = results_dir / config['output']['kmeans_results']
        kmeans_data = {k: metrics for k, (_, metrics) in kmeans_results.items()}
        save_json(kmeans_data, kmeans_file)
    
    # Hierarchical Clustering
    if config['clustering']['hierarchical']['enabled']:
        logger.info("\n--- Hierarchical Clustering ---")
        
        for linkage_method in config['clustering']['hierarchical']['linkage_methods']:
            for k in k_range:
                labels, model = pipeline.hierarchical_clustering(
                    clustering_data,
                    n_clusters=k,
                    linkage=linkage_method
                )
                
                # Evaluate
                metrics = evaluator.compute_all_metrics(clustering_data, labels)
                metrics['k'] = k
                metrics['linkage'] = linkage_method
                
                model_name = f'hierarchical_k{k}_{linkage_method}'
                all_results[model_name] = metrics
                
                logger.info(f"  k={k}, {linkage_method}: Silhouette={metrics['silhouette_score']:.4f}")
        
        # Save Hierarchical results
        hier_file = results_dir / config['output']['hierarchical_results']
        hier_data = {k: v for k, v in all_results.items() if k.startswith('hierarchical')}
        save_json(hier_data, hier_file)
    
    # Spectral Clustering
    if config['clustering']['spectral']['enabled']:
        logger.info("\n--- Spectral Clustering ---")
        
        for k in k_range:
            labels, model = pipeline.spectral_clustering(
                clustering_data,
                n_clusters=k,
                affinity=config['clustering']['spectral']['affinity'],
                gamma=config['clustering']['spectral']['gamma']
            )
            
            # Evaluate
            metrics = evaluator.compute_all_metrics(clustering_data, labels)
            metrics['k'] = k
            
            model_name = f'spectral_k{k}'
            all_results[model_name] = metrics
            
            logger.info(f"  k={k}: Silhouette={metrics['silhouette_score']:.4f}")
        
        # Save Spectral results
        spec_file = results_dir / config['output']['spectral_results']
        spec_data = {k: v for k, v in all_results.items() if k.startswith('spectral')}
        save_json(spec_data, spec_file)
    
    # ========================================================================
    # Step 5: Determine Optimal Clusters
    # ========================================================================
    logger.info("\n[5/6] Determining optimal number of clusters...")
    
    # Elbow method
    if 'kmeans_k2' in all_results:
        wcss_values = [all_results[f'kmeans_k{k}']['wcss'] for k in k_range if f'kmeans_k{k}' in all_results]
        optimal_k_elbow = evaluator.elbow_analysis(k_range, wcss_values)
        logger.info(f"Elbow method suggests: k={optimal_k_elbow}")
    
    # Silhouette analysis
    silhouette_scores = {}
    for k in k_range:
        if f'kmeans_k{k}' in all_results:
            silhouette_scores[k] = all_results[f'kmeans_k{k}']['silhouette_score']
    
    if silhouette_scores:
        optimal_k_sil = max(silhouette_scores.items(), key=lambda x: x[1])[0]
        logger.info(f"Silhouette analysis suggests: k={optimal_k_sil}")
    
    # Save optimal clusters
    optimal_clusters = {
        'elbow_method': optimal_k_elbow if 'optimal_k_elbow' in locals() else None,
        'silhouette_analysis': optimal_k_sil if 'optimal_k_sil' in locals() else None
    }
    
    optimal_file = results_dir / config['output']['optimal_clusters_file']
    save_json(optimal_clusters, optimal_file)
    
    # ========================================================================
    # Step 6: Generate Visualizations
    # ========================================================================
    logger.info("\n[6/6] Generating visualizations...")
    
    # Use optimal k for visualizations
    optimal_k = optimal_k_sil if 'optimal_k_sil' in locals() else k_range[len(k_range)//2]
    optimal_labels = pipeline.get_labels(f'kmeans_k{optimal_k}')
    
    if optimal_labels is None:
        logger.warning("Could not find labels for visualization")
    else:
        # PCA visualization
        if 'pca_2d' in config['visualization']['plots']:
            pca_viz = pipeline.apply_pca(clustering_data, n_components=2)[0]
            visualizer.plot_2d_scatter(
                pca_viz, optimal_labels,
                title=f'PCA Visualization (k={optimal_k})',
                xlabel='PC1', ylabel='PC2',
                filename='pca_clusters.png',
                show=False
            )
        
        # t-SNE visualization
        if tsne_embedding is not None and 'tsne_2d' in config['visualization']['plots']:
            visualizer.plot_2d_scatter(
                tsne_embedding, optimal_labels,
                title=f't-SNE Visualization (k={optimal_k})',
                xlabel='t-SNE 1', ylabel='t-SNE 2',
                filename='tsne_clusters.png',
                show=False
            )
        
        # UMAP visualization
        if umap_embedding is not None and 'umap_2d' in config['visualization']['plots']:
            visualizer.plot_2d_scatter(
                umap_embedding, optimal_labels,
                title=f'UMAP Visualization (k={optimal_k})',
                xlabel='UMAP 1', ylabel='UMAP 2',
                filename='umap_clusters.png',
                show=False
            )
        
        # Elbow curve
        if 'elbow_curve' in config['visualization']['plots'] and 'wcss_values' in locals():
            visualizer.plot_elbow_curve(
                k_range, wcss_values,
                optimal_k=optimal_k_elbow if 'optimal_k_elbow' in locals() else None,
                filename='elbow_curve.png',
                show=False
            )
        
        # Silhouette plot
        if 'silhouette_plot' in config['visualization']['plots']:
            avg_score = all_results[f'kmeans_k{optimal_k}']['silhouette_score']
            visualizer.plot_silhouette_analysis(
                clustering_data, optimal_labels, avg_score,
                filename=f'silhouette_k{optimal_k}.png',
                show=False
            )
        
        # Heatmap
        if 'heatmap' in config['visualization']['plots']:
            visualizer.plot_cluster_heatmap(
                data, optimal_labels,
                feature_names=feature_names,
                sample_ids=sample_ids,
                n_top_features=50,
                filename='cluster_heatmap.png',
                show=False
            )
    
    # ========================================================================
    # Step 7: Save Final Report
    # ========================================================================
    logger.info("\nGenerating final report...")
    
    # Compare results
    comparison = evaluator.compare_clustering_results(all_results)
    
    # Create report
    report = {
        'timestamp': datetime.now().isoformat(),
        'data_shape': list(data.shape),
        'clustering_data_shape': list(clustering_data.shape),
        'optimal_clusters': optimal_clusters,
        'all_results': all_results,
        'comparison': {
            'best_by_metric': comparison.get('best', {}),
            'rankings': comparison.get('rankings', {})
        },
        'config': config
    }
    
    baseline_dir = Path(config['output']['baseline_dir'])
    baseline_dir.mkdir(parents=True, exist_ok=True)
    
    report_file = baseline_dir / config['output']['baseline_report']
    save_json(report, report_file)
    
    # Save comparison table as CSV
    if all_results:
        df = pd.DataFrame(all_results).T
        csv_file = results_dir / config['output']['comparison_table']
        df.to_csv(csv_file)
        logger.info(f"Saved comparison table: {csv_file}")
    
    # ========================================================================
    # Summary
    # ========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("BASELINE CLUSTERING COMPLETE!")
    logger.info("=" * 70)
    logger.info(f"\nOptimal clusters:")
    logger.info(f"  - Elbow method: k={optimal_clusters.get('elbow_method', 'N/A')}")
    logger.info(f"  - Silhouette: k={optimal_clusters.get('silhouette_analysis', 'N/A')}")
    logger.info(f"\nResults saved to: {results_dir}")
    logger.info(f"Visualizations saved to: {viz_dir}")
    logger.info(f"Report saved to: {report_file}")
    logger.info("=" * 70)
    logger.info("\nNext step: Phase 4 - GAN Model Design & Architecture")
    logger.info("=" * 70)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Run baseline clustering experiments"
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/clustering_config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Configuration file not found: {config_path}")
        sys.exit(1)
    
    config = load_config(config_path)
    
    # Setup logging
    log_dir = Path(config['logging']['log_dir'])
    logger = setup_logging(
        log_dir=log_dir,
        log_name='baseline_clustering',
        level=getattr(__import__('logging'), args.log_level)
    )
    
    try:
        # Run clustering
        run_baseline_clustering(config, logger)
        
    except Exception as e:
        logger.error(f"Baseline clustering failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()