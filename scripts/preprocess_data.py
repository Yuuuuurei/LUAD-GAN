"""
Script to preprocess TCGA-LUAD data.
Applies complete preprocessing pipeline and saves processed data.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import argparse
import yaml
import torch
import numpy as np
from datetime import datetime

from backend.data.loader import TCGALUADLoader
from backend.data.preprocessor import LUADPreprocessor
from backend.config import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    RANDOM_SEED
)
from backend.utils import setup_logging, save_json, set_random_seeds


def load_config(config_path: Path) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def preprocess_luad_data(config: dict, logger):
    """
    Main preprocessing function.
    
    Args:
        config: Configuration dictionary
        logger: Logger instance
    """
    logger.info("=" * 70)
    logger.info("TCGA-LUAD Data Preprocessing Pipeline")
    logger.info("=" * 70)
    
    # Set random seeds
    set_random_seeds(config['reproducibility']['random_seed'])
    
    # ========================================================================
    # Step 1: Load Data
    # ========================================================================
    logger.info("\n[1/6] Loading raw data...")
    
    data_file = Path(config['input']['raw_data_file'])
    loader = TCGALUADLoader(data_file=data_file, logger=logger)
    
    # Load gene expression
    data = loader.load_gene_expression()
    
    # Filter tumor samples
    if config['sample_filtering']['keep_only_tumor']:
        data = loader.filter_tumor_samples()
    
    # Get initial statistics
    initial_stats = loader.get_data_statistics(data)
    logger.info(f"Initial data shape: {initial_stats['shape']}")
    logger.info(f"Missing values: {initial_stats['missing_values']['percentage']:.4f}%")
    
    # ========================================================================
    # Step 2: Initialize Preprocessor
    # ========================================================================
    logger.info("\n[2/6] Initializing preprocessor...")
    
    preprocessor = LUADPreprocessor(
        n_top_genes=config['feature_processing']['variance_selection']['n_top_genes'],
        min_variance=config['feature_processing']['variance_selection']['min_variance_threshold'],
        standardize=config['normalization']['standardize'],
        apply_pca=config['dimensionality_reduction']['pca']['enabled'],
        n_pca_components=config['dimensionality_reduction']['pca']['n_components'],
        random_state=config['reproducibility']['random_seed'],
        logger=logger
    )
    
    # ========================================================================
    # Step 3: Preprocess Data
    # ========================================================================
    logger.info("\n[3/6] Preprocessing data...")
    
    processed_data, metadata = preprocessor.fit_transform(
        data,
        remove_versions=config['feature_processing']['remove_ensembl_version'],
        handle_missing=config['sample_filtering']['handle_missing_values']
    )
    
    logger.info(f"\nProcessed data shape: {processed_data.shape}")
    logger.info(f"  - Samples: {processed_data.shape[0]}")
    logger.info(f"  - Features: {processed_data.shape[1]}")
    
    # ========================================================================
    # Step 4: Save Processed Data
    # ========================================================================
    logger.info("\n[4/6] Saving processed data...")
    
    output_dir = Path(config['output']['processed_data_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save as PyTorch tensor
    processed_tensor = torch.FloatTensor(processed_data)
    torch_file = output_dir / config['output']['processed_data_file']
    torch.save(processed_tensor, torch_file)
    logger.info(f"Saved PyTorch tensor: {torch_file}")
    
    # Save as NumPy array (alternative format)
    numpy_file = output_dir / config['output']['processed_data_numpy']
    np.savez_compressed(numpy_file, data=processed_data)
    logger.info(f"Saved NumPy array: {numpy_file}")
    
    # Save sample IDs
    sample_ids = data.columns.tolist()
    sample_ids_file = output_dir / config['output']['sample_ids_file']
    with open(sample_ids_file, 'w') as f:
        f.write('\n'.join(sample_ids))
    logger.info(f"Saved sample IDs: {sample_ids_file}")
    
    # ========================================================================
    # Step 5: Save Preprocessor Components
    # ========================================================================
    logger.info("\n[5/6] Saving preprocessor components...")
    
    preprocessor.save(output_dir)
    
    # ========================================================================
    # Step 6: Save Metadata and Reports
    # ========================================================================
    logger.info("\n[6/6] Saving metadata and reports...")
    
    # Create comprehensive metadata
    full_metadata = {
        'preprocessing_date': datetime.now().isoformat(),
        'config': config,
        'initial_statistics': initial_stats,
        'preprocessing_metadata': metadata,
        'final_statistics': {
            'shape': processed_data.shape,
            'n_samples': int(processed_data.shape[0]),
            'n_features': int(processed_data.shape[1]),
            'mean': float(processed_data.mean()),
            'std': float(processed_data.std()),
            'min': float(processed_data.min()),
            'max': float(processed_data.max())
        },
        'selected_genes': preprocessor.selected_genes,
        'sample_ids': sample_ids
    }
    
    # Save metadata
    metadata_file = output_dir / config['output']['metadata_file']
    save_json(full_metadata, metadata_file)
    logger.info(f"Saved metadata: {metadata_file}")
    
    # Create preprocessing report
    report = {
        'summary': {
            'original_samples': initial_stats['n_samples'],
            'original_genes': initial_stats['n_genes'],
            'final_samples': processed_data.shape[0],
            'final_features': processed_data.shape[1],
            'genes_removed': initial_stats['n_genes'] - metadata['selected_n_genes'],
            'variance_threshold': config['feature_processing']['variance_selection']['min_variance_threshold'],
            'top_n_genes': config['feature_processing']['variance_selection']['n_top_genes'],
            'standardized': config['normalization']['standardize'],
            'pca_applied': config['dimensionality_reduction']['pca']['enabled']
        },
        'data_quality': {
            'missing_values_percentage': initial_stats['missing_values']['percentage'],
            'value_range_original': initial_stats['value_range'],
            'value_range_processed': {
                'min': float(processed_data.min()),
                'max': float(processed_data.max()),
                'mean': float(processed_data.mean()),
                'std': float(processed_data.std())
            }
        }
    }
    
    if config['dimensionality_reduction']['pca']['enabled']:
        report['pca'] = {
            'n_components': config['dimensionality_reduction']['pca']['n_components'],
            'variance_explained': metadata.get('pca_variance_explained', 'N/A')
        }
    
    report_file = output_dir / config['output']['preprocessing_report']
    save_json(report, report_file)
    logger.info(f"Saved preprocessing report: {report_file}")
    
    # ========================================================================
    # Summary
    # ========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("PREPROCESSING COMPLETE!")
    logger.info("=" * 70)
    logger.info("\nSummary:")
    logger.info(f"  Original: {initial_stats['n_genes']:,} genes × {initial_stats['n_samples']} samples")
    logger.info(f"  Processed: {processed_data.shape[1]:,} features × {processed_data.shape[0]} samples")
    logger.info(f"  Reduction: {initial_stats['n_genes'] - metadata['selected_n_genes']:,} genes removed")
    logger.info(f"\nOutput files saved to: {output_dir}")
    logger.info(f"  - {config['output']['processed_data_file']}")
    logger.info(f"  - {config['output']['sample_ids_file']}")
    logger.info(f"  - {config['output']['feature_names_file']}")
    logger.info(f"  - {config['output']['metadata_file']}")
    logger.info(f"  - {config['output']['preprocessing_report']}")
    
    if config['normalization']['standardize']:
        logger.info(f"  - {config['output']['scaler_file']}")
    
    if config['dimensionality_reduction']['pca']['enabled']:
        logger.info(f"  - {config['output']['pca_transformer_file']}")
    
    logger.info("\n" + "=" * 70)
    logger.info("Next step: Phase 3 - Baseline Clustering")
    logger.info("=" * 70)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Preprocess TCGA-LUAD gene expression data"
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/data_config.yaml',
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
        log_name='preprocessing',
        level=getattr(__import__('logging'), args.log_level)
    )
    
    try:
        # Run preprocessing
        preprocess_luad_data(config, logger)
        
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()