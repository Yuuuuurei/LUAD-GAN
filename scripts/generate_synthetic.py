"""
Script to generate synthetic samples using trained GAN.
Creates augmented dataset for clustering.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import argparse
import torch
import numpy as np
import yaml

from backend.models import WGAN_GP, create_wgan_gp_from_config
from backend.data.augmentation import DataAugmenter
from backend.validation import validate_synthetic_data
from backend.config import PROCESSED_DATA_DIR
from backend.utils import setup_logging, set_random_seeds, save_json


def main():
    """Main generation function."""
    parser = argparse.ArgumentParser(description="Generate synthetic samples")
    parser.add_argument('--config', type=str, default='configs/gan_config.yaml')
    parser.add_argument('--model', type=str, default='models/checkpoints/wgan_gp_best.pt')
    parser.add_argument('--augmentation-ratio', type=float, default=1.0)
    parser.add_argument('--strategy', type=str, default='combine', choices=['combine', 'synthetic_only', 'mixed'])
    parser.add_argument('--output-dir', type=str, default='data/synthetic')
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup logging
    log_dir = Path('logs/generation')
    logger = setup_logging(log_dir, 'synthetic_generation')
    
    logger.info("="*70)
    logger.info("Synthetic Data Generation")
    logger.info("="*70)
    
    # Set seeds
    set_random_seeds(config['reproducibility']['random_seed'])
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")
    
    # Load real data
    data_file = PROCESSED_DATA_DIR / "luad_processed.pt"
    real_data = torch.load(data_file)
    logger.info(f"Loaded real data: {real_data.shape}")
    
    # Load model
    model = create_wgan_gp_from_config(config, num_features=real_data.shape[1])
    checkpoint = torch.load(args.model, map_location=device)
    model.generator.load_state_dict(checkpoint['generator_state_dict'])
    model = model.to(device)
    model.eval()
    logger.info(f"Loaded model from: {args.model}")
    
    # Generate synthetic samples
    n_synthetic = int(real_data.shape[0] * args.augmentation_ratio)
    logger.info(f"\nGenerating {n_synthetic} synthetic samples...")
    
    with torch.no_grad():
        z = torch.randn(n_synthetic, model.latent_dim, device=device)
        synthetic_data = model.generator(z).cpu()
    
    logger.info(f"Generated synthetic data: {synthetic_data.shape}")
    
    # Validate quality
    logger.info("\nValidating synthetic data quality...")
    metrics = validate_synthetic_data(
        real_data.numpy(),
        synthetic_data.numpy()
    )
    
    # Save synthetic data
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    synthetic_file = output_dir / "gan_generated_samples.pt"
    torch.save(synthetic_data, synthetic_file)
    logger.info(f"\nSaved synthetic data: {synthetic_file}")
    
    # Create augmented dataset
    logger.info(f"\nCreating augmented dataset (strategy: {args.strategy})...")
    augmenter = DataAugmenter(
        augmentation_ratio=args.augmentation_ratio,
        strategy=args.strategy,
        logger=logger
    )
    
    augmented_data, labels = augmenter.augment(real_data, synthetic_data)
    
    # Save augmented dataset
    augmenter.save(output_dir)
    
    # Save quality report
    quality_report = {
        'augmentation_ratio': args.augmentation_ratio,
        'strategy': args.strategy,
        'n_real': int((labels == 0).sum()),
        'n_synthetic': int((labels == 1).sum()),
        'total_samples': len(labels),
        'quality_metrics': metrics
    }
    
    report_file = Path('results/gan_validation/generation_report.json')
    report_file.parent.mkdir(parents=True, exist_ok=True)
    save_json(quality_report, report_file)
    logger.info(f"Saved quality report: {report_file}")
    
    logger.info("\n" + "="*70)
    logger.info("Generation Complete!")
    logger.info("="*70)
    logger.info(f"Real samples: {(labels == 0).sum()}")
    logger.info(f"Synthetic samples: {(labels == 1).sum()}")
    logger.info(f"Total samples: {len(labels)}")
    logger.info(f"Quality score: {metrics['quality_score']}")
    logger.info("="*70)


if __name__ == "__main__":
    main()