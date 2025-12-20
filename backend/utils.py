"""
Utility functions for GAN-LUAD Clustering project.
"""

import random
import numpy as np
import torch
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple
import json

def set_random_seeds(seed: int = 42):
    """
    Set random seeds for reproducibility across numpy, random, and torch.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Additional CUDA settings for reproducibility
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    print(f"Random seeds set to {seed}")

def setup_logging(
    log_dir: Path,
    log_name: str = "experiment",
    level: int = logging.INFO
) -> logging.Logger:
    """
    Set up logging configuration.
    
    Args:
        log_dir: Directory to save log files
        log_name: Name prefix for log file
        level: Logging level
        
    Returns:
        Configured logger
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"{log_name}_{timestamp}.log"
    
    # Create logger
    logger = logging.getLogger(log_name)
    logger.setLevel(level)
    
    # Remove existing handlers
    logger.handlers.clear()
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    logger.info(f"Logging initialized. Log file: {log_file}")
    
    return logger

def save_json(data: dict, filepath: Path):
    """
    Save dictionary to JSON file.
    
    Args:
        data: Dictionary to save
        filepath: Path to save JSON file
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    def convert_numpy(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2, default=convert_numpy)
    print(f"Saved JSON to {filepath}")

def load_json(filepath: Path) -> dict:
    """
    Load JSON file to dictionary.
    
    Args:
        filepath: Path to JSON file
        
    Returns:
        Loaded dictionary
    """
    with open(filepath, 'r') as f:
        data = json.load(f)
    print(f"Loaded JSON from {filepath}")
    return data

def get_device(gpu_id: Optional[int] = None) -> torch.device:
    """
    Get torch device (GPU if available, else CPU).
    
    Args:
        gpu_id: Specific GPU ID to use (None for auto-select)
        
    Returns:
        torch.device
    """
    if torch.cuda.is_available():
        if gpu_id is not None:
            device = torch.device(f"cuda:{gpu_id}")
        else:
            device = torch.device("cuda")
        print(f"Using device: {device} ({torch.cuda.get_device_name(device)})")
    else:
        device = torch.device("cpu")
        print("CUDA not available. Using CPU.")
    
    return device

def format_time(seconds: float) -> str:
    """
    Format seconds into readable time string.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string (e.g., "1h 23m 45s")
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"

def count_parameters(model: torch.nn.Module) -> int:
    """
    Count total number of trainable parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def print_model_summary(model: torch.nn.Module, model_name: str = "Model"):
    """
    Print model architecture summary.
    
    Args:
        model: PyTorch model
        model_name: Name of the model
    """
    print(f"\n{'='*60}")
    print(f"{model_name} Architecture")
    print(f"{'='*60}")
    print(model)
    print(f"{'='*60}")
    print(f"Total trainable parameters: {count_parameters(model):,}")
    print(f"{'='*60}\n")

def check_nan_inf(tensor: torch.Tensor, name: str = "tensor") -> bool:
    """
    Check if tensor contains NaN or Inf values.
    
    Args:
        tensor: Input tensor
        name: Name for logging
        
    Returns:
        True if NaN or Inf found, False otherwise
    """
    has_nan = torch.isnan(tensor).any().item()
    has_inf = torch.isinf(tensor).any().item()
    
    if has_nan or has_inf:
        print(f"WARNING: {name} contains NaN: {has_nan}, Inf: {has_inf}")
        return True
    return False

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    filepath: Path,
    additional_info: Optional[dict] = None
):
    """
    Save model checkpoint.
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        epoch: Current epoch
        loss: Current loss value
        filepath: Path to save checkpoint
        additional_info: Additional information to save
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'timestamp': datetime.now().isoformat()
    }
    
    if additional_info:
        checkpoint.update(additional_info)
    
    filepath.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to {filepath}")

def load_checkpoint(
    filepath: Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: Optional[torch.device] = None
) -> Tuple[int, float]:
    """
    Load model checkpoint.
    
    Args:
        filepath: Path to checkpoint file
        model: PyTorch model
        optimizer: Optimizer (optional)
        device: Device to load model to
        
    Returns:
        Tuple of (epoch, loss)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint.get('epoch', 0)
    loss = checkpoint.get('loss', float('inf'))
    
    print(f"Checkpoint loaded from {filepath} (Epoch: {epoch}, Loss: {loss:.4f})")
    
    return epoch, loss

class EarlyStopping:
    """
    Early stopping to stop training when loss doesn't improve.
    """
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0, mode: str = 'min'):
        """
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'min' for loss (lower is better), 'max' for accuracy
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score: float) -> bool:
        """
        Check if training should stop.
        
        Args:
            score: Current metric value
            
        Returns:
            True if should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = score
        elif self._is_improvement(score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop
    
    def _is_improvement(self, score: float) -> bool:
        """Check if score is an improvement."""
        if self.mode == 'min':
            return score < (self.best_score - self.min_delta)
        else:
            return score > (self.best_score + self.min_delta)

if __name__ == "__main__":
    # Test utilities
    print("Testing utility functions...")
    set_random_seeds(42)
    device = get_device()
    print(f"Time formatting test: {format_time(3725)}")
    print("Utilities working correctly!")