"""
Training callbacks for monitoring and control.
"""

import torch
import numpy as np
from pathlib import Path
from typing import Optional, Callable
import logging


class EarlyStopping:
    """
    Early stopping to stop training when monitored metric doesn't improve.
    """
    
    def __init__(
        self,
        patience: int = 50,
        min_delta: float = 0.001,
        mode: str = 'min',
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait
            min_delta: Minimum change to qualify as improvement
            mode: 'min' for loss (lower is better), 'max' for score
            logger: Logger instance
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.logger = logger or logging.getLogger(__name__)
        
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0
    
    def __call__(self, score: float, epoch: int) -> bool:
        """
        Check if training should stop.
        
        Args:
            score: Current metric value
            epoch: Current epoch
            
        Returns:
            True if should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
        elif self._is_improvement(score):
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                self.logger.info(
                    f"Early stopping triggered after {epoch} epochs. "
                    f"Best epoch: {self.best_epoch} with score: {self.best_score:.6f}"
                )
        
        return self.early_stop
    
    def _is_improvement(self, score: float) -> bool:
        """Check if score is an improvement."""
        if self.mode == 'min':
            return score < (self.best_score - self.min_delta)
        else:
            return score > (self.best_score + self.min_delta)
    
    def reset(self):
        """Reset early stopping state."""
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0


class ModelCheckpoint:
    """
    Save model checkpoints during training.
    """
    
    def __init__(
        self,
        checkpoint_dir: Path,
        save_best: bool = True,
        monitor: str = 'critic_loss',
        mode: str = 'min',
        save_interval: int = 50,
        keep_last_n: int = 3,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize model checkpoint.
        
        Args:
            checkpoint_dir: Directory to save checkpoints
            save_best: Whether to save best model
            monitor: Metric to monitor
            mode: 'min' or 'max'
            save_interval: Save every N epochs
            keep_last_n: Keep only last N checkpoints (0 = keep all)
            logger: Logger instance
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.save_best = save_best
        self.monitor = monitor
        self.mode = mode
        self.save_interval = save_interval
        self.keep_last_n = keep_last_n
        self.logger = logger or logging.getLogger(__name__)
        
        self.best_score = None
        self.saved_checkpoints = []
    
    def __call__(
        self,
        model,
        optimizer_g,
        optimizer_c,
        epoch: int,
        metrics: dict
    ):
        """
        Save checkpoint if conditions are met.
        
        Args:
            model: Model to save
            optimizer_g: Generator optimizer
            optimizer_c: Critic optimizer
            epoch: Current epoch
            metrics: Dictionary of metrics
        """
        # Save at intervals
        if epoch % self.save_interval == 0:
            self._save_checkpoint(
                model, optimizer_g, optimizer_c, epoch, metrics,
                filename=f'wgan_gp_epoch_{epoch}.pt'
            )
        
        # Save best model
        if self.save_best and self.monitor in metrics:
            score = metrics[self.monitor]
            
            if self._is_best(score):
                self.best_score = score
                self._save_checkpoint(
                    model, optimizer_g, optimizer_c, epoch, metrics,
                    filename='wgan_gp_best.pt'
                )
                self.logger.info(f"New best model saved at epoch {epoch} with {self.monitor}={score:.6f}")
    
    def _is_best(self, score: float) -> bool:
        """Check if score is the best."""
        if self.best_score is None:
            return True
        
        if self.mode == 'min':
            return score < self.best_score
        else:
            return score > self.best_score
    
    def _save_checkpoint(
        self,
        model,
        optimizer_g,
        optimizer_c,
        epoch: int,
        metrics: dict,
        filename: str
    ):
        """Save checkpoint to file."""
        filepath = self.checkpoint_dir / filename
        
        checkpoint = {
            'epoch': epoch,
            'generator_state_dict': model.generator.state_dict(),
            'critic_state_dict': model.critic.state_dict(),
            'optimizer_g_state_dict': optimizer_g.state_dict(),
            'optimizer_c_state_dict': optimizer_c.state_dict(),
            'metrics': metrics,
            'num_features': model.num_features,
            'latent_dim': model.latent_dim
        }
        
        torch.save(checkpoint, filepath)
        
        # Track saved checkpoints
        if filename.startswith('wgan_gp_epoch_'):
            self.saved_checkpoints.append(filepath)
            self._cleanup_old_checkpoints()
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints to save space."""
        if self.keep_last_n > 0 and len(self.saved_checkpoints) > self.keep_last_n:
            # Sort by epoch number
            self.saved_checkpoints.sort(key=lambda x: int(x.stem.split('_')[-1]))
            
            # Remove oldest
            to_remove = self.saved_checkpoints[:-self.keep_last_n]
            for filepath in to_remove:
                if filepath.exists():
                    filepath.unlink()
                    self.logger.debug(f"Removed old checkpoint: {filepath}")
            
            self.saved_checkpoints = self.saved_checkpoints[-self.keep_last_n:]


class LossHistory:
    """
    Track and save loss history during training.
    """
    
    def __init__(self):
        """Initialize loss history."""
        self.history = {
            'epoch': [],
            'generator_loss': [],
            'critic_loss': [],
            'wasserstein_distance': [],
            'gradient_penalty': []
        }
    
    def update(
        self,
        epoch: int,
        generator_loss: float,
        critic_loss: float,
        wasserstein_distance: float,
        gradient_penalty: float
    ):
        """
        Update history with new values.
        
        Args:
            epoch: Current epoch
            generator_loss: Generator loss
            critic_loss: Critic loss
            wasserstein_distance: Wasserstein distance
            gradient_penalty: Gradient penalty
        """
        self.history['epoch'].append(epoch)
        self.history['generator_loss'].append(generator_loss)
        self.history['critic_loss'].append(critic_loss)
        self.history['wasserstein_distance'].append(wasserstein_distance)
        self.history['gradient_penalty'].append(gradient_penalty)
    
    def get_last(self, key: str, n: int = 1) -> float:
        """
        Get last n values for a metric.
        
        Args:
            key: Metric name
            n: Number of values to get
            
        Returns:
            Last n values (or average if n > 1)
        """
        if key not in self.history or not self.history[key]:
            return 0.0
        
        values = self.history[key][-n:]
        return np.mean(values)
    
    def save(self, filepath: Path):
        """Save history to JSON file."""
        import json
        with open(filepath, 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def load(self, filepath: Path):
        """Load history from JSON file."""
        import json
        with open(filepath, 'r') as f:
            self.history = json.load(f)


class ProgressCallback:
    """
    Display training progress.
    """
    
    def __init__(
        self,
        total_epochs: int,
        print_interval: int = 1,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize progress callback.
        
        Args:
            total_epochs: Total number of epochs
            print_interval: Print every N epochs
            logger: Logger instance
        """
        self.total_epochs = total_epochs
        self.print_interval = print_interval
        self.logger = logger or logging.getLogger(__name__)
    
    def __call__(
        self,
        epoch: int,
        metrics: dict,
        elapsed_time: float
    ):
        """
        Print progress.
        
        Args:
            epoch: Current epoch
            metrics: Dictionary of metrics
            elapsed_time: Elapsed time in seconds
        """
        if epoch % self.print_interval == 0:
            progress = (epoch / self.total_epochs) * 100
            
            msg = f"Epoch [{epoch}/{self.total_epochs}] ({progress:.1f}%) | "
            msg += f"Time: {elapsed_time:.2f}s | "
            
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    msg += f"{key}: {value:.6f} | "
            
            self.logger.info(msg.rstrip(" | "))


if __name__ == "__main__":
    print("Testing callbacks...")
    
    # Test early stopping
    early_stopping = EarlyStopping(patience=5, mode='min')
    
    scores = [1.0, 0.9, 0.85, 0.84, 0.83, 0.83, 0.83, 0.83, 0.83, 0.83]
    for epoch, score in enumerate(scores, 1):
        stop = early_stopping(score, epoch)
        print(f"Epoch {epoch}: score={score:.2f}, stop={stop}")
        if stop:
            break
    
    print("\n✓ Callbacks test passed!")