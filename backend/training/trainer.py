"""
Trainer for WGAN-GP.
Handles the complete training loop with monitoring and checkpointing.
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import time
from pathlib import Path
from typing import Optional, Dict
import logging

from backend.models import WGAN_GP
from .callbacks import EarlyStopping, ModelCheckpoint, LossHistory, ProgressCallback
from backend.utils import set_random_seeds


class WGANTrainer:
    """
    Trainer for WGAN-GP model.
    """
    
    def __init__(
        self,
        model: WGAN_GP,
        config: dict,
        device: torch.device,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize trainer.
        
        Args:
            model: WGAN-GP model
            config: Training configuration
            device: Device to train on
            logger: Logger instance
        """
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.logger = logger or logging.getLogger(__name__)
        
        # Training parameters
        self.num_epochs = config['training']['num_epochs']
        self.batch_size = config['training']['batch_size']
        self.n_critic = config['training']['n_critic']
        self.noise_std = config['training']['regularization'].get('noise_std', 0.0)
        
        # Initialize optimizers
        self._init_optimizers()
        
        # Initialize callbacks
        self._init_callbacks()
        
        # Loss history
        self.loss_history = LossHistory()
        
        # Current epoch
        self.current_epoch = 0
    
    def _init_optimizers(self):
        """Initialize optimizers for generator and critic."""
        gen_config = self.config['training']['optimizer']['generator']
        critic_config = self.config['training']['optimizer']['critic']
        
        # Generator optimizer
        if gen_config['type'] == 'adam':
            self.optimizer_g = optim.Adam(
                self.model.generator.parameters(),
                lr=gen_config['learning_rate'],
                betas=gen_config['betas'],
                weight_decay=gen_config['weight_decay']
            )
        else:
            raise ValueError(f"Unknown optimizer: {gen_config['type']}")
        
        # Critic optimizer
        if critic_config['type'] == 'adam':
            self.optimizer_c = optim.Adam(
                self.model.critic.parameters(),
                lr=critic_config['learning_rate'],
                betas=critic_config['betas'],
                weight_decay=critic_config['weight_decay']
            )
        else:
            raise ValueError(f"Unknown optimizer: {critic_config['type']}")
        
        self.logger.info("Optimizers initialized")
    
    def _init_callbacks(self):
        """Initialize training callbacks."""
        # Early stopping
        if self.config['monitoring']['early_stopping']['enabled']:
            self.early_stopping = EarlyStopping(
                patience=self.config['monitoring']['early_stopping']['patience'],
                min_delta=self.config['monitoring']['early_stopping']['min_delta'],
                mode='min',
                logger=self.logger
            )
        else:
            self.early_stopping = None
        
        # Model checkpoint
        checkpoint_dir = Path(self.config['output']['checkpoint_dir'])
        self.model_checkpoint = ModelCheckpoint(
            checkpoint_dir=checkpoint_dir,
            save_best=self.config['monitoring']['save_best'],
            monitor='critic_loss',
            mode='min',
            save_interval=self.config['monitoring']['checkpoint_interval'],
            keep_last_n=self.config['monitoring']['keep_last_n'],
            logger=self.logger
        )
        
        # Progress callback
        self.progress_callback = ProgressCallback(
            total_epochs=self.num_epochs,
            print_interval=self.config['monitoring']['print_interval'],
            logger=self.logger
        )
        
        self.logger.info("Callbacks initialized")
    
    def train(self, data: torch.Tensor) -> Dict:
        """
        Train the model.
        
        Args:
            data: Training data (samples × features)
            
        Returns:
            Dictionary with training results
        """
        self.logger.info("=" * 70)
        self.logger.info("Starting WGAN-GP Training")
        self.logger.info("=" * 70)
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Training samples: {data.shape[0]}")
        self.logger.info(f"Features: {data.shape[1]}")
        self.logger.info(f"Epochs: {self.num_epochs}")
        self.logger.info(f"Batch size: {self.batch_size}")
        self.logger.info(f"N-critic: {self.n_critic}")
        
        # Create data loader
        dataset = TensorDataset(data)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.config['hardware']['num_workers'],
            pin_memory=self.config['hardware']['pin_memory']
        )
        
        # Training loop
        start_time = time.time()
        
        for epoch in range(1, self.num_epochs + 1):
            self.current_epoch = epoch
            epoch_start = time.time()
            
            # Train one epoch
            metrics = self._train_epoch(dataloader)
            
            # Update history
            self.loss_history.update(
                epoch=epoch,
                generator_loss=metrics['generator_loss'],
                critic_loss=metrics['critic_loss'],
                wasserstein_distance=metrics['wasserstein_distance'],
                gradient_penalty=metrics['gradient_penalty']
            )
            
            # Callbacks
            epoch_time = time.time() - epoch_start
            self.progress_callback(epoch, metrics, epoch_time)
            
            # Save checkpoint
            self.model_checkpoint(
                self.model,
                self.optimizer_g,
                self.optimizer_c,
                epoch,
                metrics
            )
            
            # Early stopping
            if self.early_stopping:
                if self.early_stopping(metrics['critic_loss'], epoch):
                    self.logger.info("Early stopping triggered")
                    break
        
        # Training complete
        total_time = time.time() - start_time
        self.logger.info("=" * 70)
        self.logger.info(f"Training complete! Total time: {total_time:.2f}s")
        self.logger.info("=" * 70)
        
        # Save final model
        final_path = Path(self.config['output']['checkpoint_dir']) / self.config['output']['final_model_name']
        self.model.save(str(final_path))
        self.logger.info(f"Final model saved: {final_path}")
        
        # Save loss history
        history_path = Path(self.config['output']['log_dir']) / self.config['output']['loss_history']
        history_path.parent.mkdir(parents=True, exist_ok=True)
        self.loss_history.save(history_path)
        self.logger.info(f"Loss history saved: {history_path}")
        
        return {
            'total_epochs': epoch,
            'total_time': total_time,
            'final_metrics': metrics,
            'history': self.loss_history.history
        }
    
    def _train_epoch(self, dataloader: DataLoader) -> Dict:
        """
        Train one epoch.
        
        Args:
            dataloader: Data loader
            
        Returns:
            Dictionary of metrics
        """
        self.model.train()
        
        generator_losses = []
        critic_losses = []
        wasserstein_distances = []
        gradient_penalties = []
        
        for batch_idx, (real_data,) in enumerate(dataloader):
            real_data = real_data.to(self.device)
            batch_size = real_data.size(0)
            
            # Add instance noise if enabled
            if self.noise_std > 0:
                noise = torch.randn_like(real_data) * self.noise_std
                real_data = real_data + noise
            
            # Train Critic
            for _ in range(self.n_critic):
                self.optimizer_c.zero_grad()
                
                # Generate fake samples
                z = torch.randn(batch_size, self.model.latent_dim, device=self.device)
                fake_data = self.model.generator(z)
                
                # Add instance noise to fake samples too
                if self.noise_std > 0:
                    noise = torch.randn_like(fake_data) * self.noise_std
                    fake_data = fake_data + noise
                
                # Compute critic loss
                critic_loss, wd, gp = self.model.compute_critic_loss(
                    real_data,
                    fake_data.detach(),
                    self.device
                )
                
                # Backward and optimize
                critic_loss.backward()
                self.optimizer_c.step()
                
                critic_losses.append(critic_loss.item())
                wasserstein_distances.append(wd.item())
                gradient_penalties.append(gp.item())
            
            # Train Generator
            self.optimizer_g.zero_grad()
            
            # Generate fake samples
            z = torch.randn(batch_size, self.model.latent_dim, device=self.device)
            fake_data = self.model.generator(z)
            
            # Add instance noise to fake samples
            if self.noise_std > 0:
                noise = torch.randn_like(fake_data) * self.noise_std
                fake_data = fake_data + noise
            
            # Compute generator loss (with variance regularization if enabled)
            generator_loss = self.model.compute_generator_loss(fake_data, real_data)
            
            # Backward and optimize
            generator_loss.backward()
            self.optimizer_g.step()
            
            generator_losses.append(generator_loss.item())
        
        # Average metrics
        metrics = {
            'generator_loss': sum(generator_losses) / len(generator_losses),
            'critic_loss': sum(critic_losses) / len(critic_losses),
            'wasserstein_distance': sum(wasserstein_distances) / len(wasserstein_distances),
            'gradient_penalty': sum(gradient_penalties) / len(gradient_penalties)
        }
        
        return metrics
    
    def generate_samples(self, n_samples: int) -> torch.Tensor:
        """
        Generate synthetic samples.
        
        Args:
            n_samples: Number of samples to generate
            
        Returns:
            Generated samples
        """
        self.model.eval()
        with torch.no_grad():
            samples = self.model.generate(n_samples, self.device)
        return samples.cpu()


if __name__ == "__main__":
    print("Testing WGANTrainer...")
    
    # Create dummy model and data
    from backend.models import WGAN_GP
    
    model = WGAN_GP(num_features=100, latent_dim=32)
    data = torch.randn(200, 100)
    device = torch.device('cpu')
    
    # Dummy config
    config = {
        'training': {
            'num_epochs': 5,
            'batch_size': 32,
            'n_critic': 2,
            'optimizer': {
                'generator': {
                    'type': 'adam',
                    'learning_rate': 0.0001,
                    'betas': [0.5, 0.9],
                    'weight_decay': 0.0
                },
                'critic': {
                    'type': 'adam',
                    'learning_rate': 0.0001,
                    'betas': [0.5, 0.9],
                    'weight_decay': 0.0
                }
            },
            'wgan_gp': {
                'gradient_penalty_weight': 10
            }
        },
        'monitoring': {
            'early_stopping': {'enabled': False},
            'save_best': False,
            'checkpoint_interval': 10,
            'keep_last_n': 2,
            'print_interval': 1
        },
        'output': {
            'checkpoint_dir': 'test_checkpoints',
            'log_dir': 'test_logs',
            'final_model_name': 'test_final.pt',
            'loss_history': 'test_history.json'
        },
        'hardware': {
            'num_workers': 0,
            'pin_memory': False
        }
    }
    
    # Train
    trainer = WGANTrainer(model, config, device)
    results = trainer.train(data)
    
    print(f"\nTraining results:")
    print(f"  Total epochs: {results['total_epochs']}")
    print(f"  Total time: {results['total_time']:.2f}s")
    
    # Cleanup
    import shutil
    shutil.rmtree('test_checkpoints', ignore_errors=True)
    shutil.rmtree('test_logs', ignore_errors=True)
    
    print("\n✓ WGANTrainer test passed!")