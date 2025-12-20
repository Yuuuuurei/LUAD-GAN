"""
Script to train WGAN-GP on TCGA-LUAD data.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import argparse
import yaml
import torch

from backend.models import create_wgan_gp_from_config
from backend.training import WGANTrainer
from backend.config import PROCESSED_DATA_DIR
from backend.utils import setup_logging, set_random_seeds


def load_config(config_path: Path) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train WGAN-GP")
    parser.add_argument('--config', type=str, default='configs/gan_config.yaml')
    parser.add_argument('--log-level', type=str, default='INFO')
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(Path(args.config))
    
    # Setup logging
    log_dir = Path(config['output']['log_dir'])
    logger = setup_logging(log_dir, 'gan_training', level=getattr(__import__('logging'), args.log_level))
    
    logger.info("="*70)
    logger.info("WGAN-GP Training Script")
    logger.info("="*70)
    
    # Set random seeds
    set_random_seeds(config['reproducibility']['random_seed'])
    
    # Device
    if config['hardware']['device'] == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(config['hardware']['device'])
    
    logger.info(f"Device: {device}")
    
    # Load data
    data_file = Path(config['input']['processed_data_file'])
    data_tensor = torch.load(data_file)
    data = data_tensor.to(device)
    
    logger.info(f"Loaded data: {data.shape}")
    
    # Create model
    model = create_wgan_gp_from_config(config, num_features=data.shape[1])
    logger.info(f"Model created")
    
    total, gen, critic = model.count_parameters()
    logger.info(f"Parameters: Total={total:,}, Generator={gen:,}, Critic={critic:,}")
    
    # Create trainer
    trainer = WGANTrainer(model, config, device, logger)
    
    # Train
    results = trainer.train(data)
    
    logger.info("="*70)
    logger.info("Training Complete!")
    logger.info(f"Total epochs: {results['total_epochs']}")
    logger.info(f"Total time: {results['total_time']:.2f}s")
    logger.info("="*70)


if __name__ == "__main__":
    main()