"""
Data augmentation module for GAN-generated synthetic data.
Combines real and synthetic samples for clustering.
"""

import torch
import numpy as np
from typing import Tuple, Optional, Dict
from pathlib import Path
import logging


class DataAugmenter:
    """
    Augmenter for combining real and synthetic data.
    """
    
    def __init__(
        self,
        augmentation_ratio: float = 1.0,
        strategy: str = "combine",
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize data augmenter.
        
        Args:
            augmentation_ratio: Ratio of synthetic to real data (1.0 = same size)
            strategy: Augmentation strategy (combine, synthetic_only, mixed)
            logger: Logger instance
        """
        self.augmentation_ratio = augmentation_ratio
        self.strategy = strategy
        self.logger = logger or logging.getLogger(__name__)
        
        self.real_data = None
        self.synthetic_data = None
        self.augmented_data = None
        self.labels = None  # 0 = real, 1 = synthetic
    
    def augment(
        self,
        real_data: torch.Tensor,
        synthetic_data: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Augment real data with synthetic data.
        
        Args:
            real_data: Real data samples (n_real × features)
            synthetic_data: Synthetic data samples (n_synthetic × features)
            
        Returns:
            Tuple of (augmented_data, labels)
        """
        self.real_data = real_data
        self.synthetic_data = synthetic_data
        
        if self.strategy == "combine":
            # Strategy 1: Add synthetic to real
            self.augmented_data, self.labels = self._combine_strategy()
            
        elif self.strategy == "synthetic_only":
            # Strategy 2: Use only synthetic
            self.augmented_data, self.labels = self._synthetic_only_strategy()
            
        elif self.strategy == "mixed":
            # Strategy 3: Mixed approach (50/50)
            self.augmented_data, self.labels = self._mixed_strategy()
            
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
        
        self.logger.info(f"Augmentation complete using '{self.strategy}' strategy")
        self.logger.info(f"  Real samples: {(self.labels == 0).sum().item()}")
        self.logger.info(f"  Synthetic samples: {(self.labels == 1).sum().item()}")
        self.logger.info(f"  Total samples: {len(self.labels)}")
        
        return self.augmented_data, self.labels
    
    def _combine_strategy(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Strategy 1: Combine all real + synthetic data.
        
        Returns:
            Tuple of (augmented_data, labels)
        """
        # Determine number of synthetic samples to add
        n_real = self.real_data.shape[0]
        n_synthetic_needed = int(n_real * self.augmentation_ratio)
        
        # Sample synthetic data if we have more than needed
        if n_synthetic_needed < self.synthetic_data.shape[0]:
            indices = torch.randperm(self.synthetic_data.shape[0])[:n_synthetic_needed]
            synthetic_subset = self.synthetic_data[indices]
        else:
            synthetic_subset = self.synthetic_data[:n_synthetic_needed]
        
        # Combine
        augmented_data = torch.cat([self.real_data, synthetic_subset], dim=0)
        
        # Create labels
        labels = torch.cat([
            torch.zeros(n_real, dtype=torch.long),
            torch.ones(synthetic_subset.shape[0], dtype=torch.long)
        ])
        
        return augmented_data, labels
    
    def _synthetic_only_strategy(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Strategy 2: Use only synthetic data.
        
        Returns:
            Tuple of (augmented_data, labels)
        """
        # Use all synthetic data
        augmented_data = self.synthetic_data
        labels = torch.ones(self.synthetic_data.shape[0], dtype=torch.long)
        
        return augmented_data, labels
    
    def _mixed_strategy(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Strategy 3: Mixed approach (equal parts real and synthetic).
        
        Returns:
            Tuple of (augmented_data, labels)
        """
        n_real = self.real_data.shape[0]
        n_synthetic = min(n_real, self.synthetic_data.shape[0])
        
        # Sample equal amounts
        real_indices = torch.randperm(n_real)[:n_synthetic]
        synthetic_indices = torch.randperm(self.synthetic_data.shape[0])[:n_synthetic]
        
        real_subset = self.real_data[real_indices]
        synthetic_subset = self.synthetic_data[synthetic_indices]
        
        # Combine
        augmented_data = torch.cat([real_subset, synthetic_subset], dim=0)
        labels = torch.cat([
            torch.zeros(n_synthetic, dtype=torch.long),
            torch.ones(n_synthetic, dtype=torch.long)
        ])
        
        return augmented_data, labels
    
    def save(self, output_dir: Path):
        """
        Save augmented data and labels.
        
        Args:
            output_dir: Directory to save data
        """
        if self.augmented_data is None:
            raise ValueError("No augmented data to save. Call augment() first.")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save augmented data
        augmented_file = output_dir / "augmented_data.pt"
        torch.save(self.augmented_data, augmented_file)
        self.logger.info(f"Saved augmented data: {augmented_file}")
        
        # Save labels
        labels_file = output_dir / "augmented_labels.pt"
        torch.save(self.labels, labels_file)
        self.logger.info(f"Saved labels: {labels_file}")
        
        # Save metadata
        metadata = {
            'strategy': self.strategy,
            'augmentation_ratio': self.augmentation_ratio,
            'n_real': int((self.labels == 0).sum()),
            'n_synthetic': int((self.labels == 1).sum()),
            'total_samples': len(self.labels),
            'n_features': self.augmented_data.shape[1]
        }
        
        import json
        metadata_file = output_dir / "augmentation_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        self.logger.info(f"Saved metadata: {metadata_file}")
        
        # Save as .npz for compatibility
        npz_file = output_dir / "augmented_data.npz"
        np.savez(npz_file, data=self.augmented_data.numpy(), labels=self.labels.numpy())
        self.logger.info(f"Saved .npz file: {npz_file}")
    
    def load(self, output_dir: Path):
        """
        Load augmented data and labels.
        
        Args:
            output_dir: Directory containing saved data
        """
        output_dir = Path(output_dir)
        
        # Load augmented data
        augmented_file = output_dir / "augmented_data.pt"
        self.augmented_data = torch.load(augmented_file)
        self.logger.info(f"Loaded augmented data: {augmented_file}")
        
        # Load labels
        labels_file = output_dir / "augmented_labels.pt"
        self.labels = torch.load(labels_file)
        self.logger.info(f"Loaded labels: {labels_file}")
        
        # Load metadata
        import json
        metadata_file = output_dir / "augmentation_metadata.json"
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            self.strategy = metadata['strategy']
            self.augmentation_ratio = metadata['augmentation_ratio']
            self.logger.info(f"Loaded metadata: {metadata_file}")


def create_augmented_dataset(
    real_data: torch.Tensor,
    synthetic_data: torch.Tensor,
    augmentation_ratio: float = 1.0,
    strategy: str = "combine"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convenience function to create augmented dataset.
    
    Args:
        real_data: Real data samples
        synthetic_data: Synthetic data samples
        augmentation_ratio: Ratio of synthetic to real
        strategy: Augmentation strategy
        
    Returns:
        Tuple of (augmented_data, labels)
    """
    augmenter = DataAugmenter(augmentation_ratio=augmentation_ratio, strategy=strategy)
    augmented_data, labels = augmenter.augment(real_data, synthetic_data)
    
    return augmented_data, labels


def test_augmentation_strategies(
    real_data: torch.Tensor,
    synthetic_data: torch.Tensor
) -> Dict[str, Dict]:
    """
    Test all augmentation strategies and return results.
    
    Args:
        real_data: Real data samples
        synthetic_data: Synthetic data samples
        
    Returns:
        Dictionary with results for each strategy
    """
    strategies = ["combine", "synthetic_only", "mixed"]
    results = {}
    
    for strategy in strategies:
        augmenter = DataAugmenter(augmentation_ratio=1.0, strategy=strategy)
        augmented_data, labels = augmenter.augment(real_data, synthetic_data)
        
        results[strategy] = {
            'total_samples': augmented_data.shape[0],
            'n_real': int((labels == 0).sum()),
            'n_synthetic': int((labels == 1).sum()),
            'shape': augmented_data.shape
        }
    
    return results


if __name__ == "__main__":
    print("Testing DataAugmenter...")
    
    # Create dummy data
    real_data = torch.randn(100, 50)
    synthetic_data = torch.randn(150, 50)
    
    # Test combine strategy
    augmenter = DataAugmenter(augmentation_ratio=1.0, strategy="combine")
    augmented, labels = augmenter.augment(real_data, synthetic_data)
    
    print(f"\nCombine strategy:")
    print(f"  Real: {(labels == 0).sum()}")
    print(f"  Synthetic: {(labels == 1).sum()}")
    print(f"  Total: {len(labels)}")
    
    # Test all strategies
    results = test_augmentation_strategies(real_data, synthetic_data)
    print(f"\nAll strategies:")
    for strategy, info in results.items():
        print(f"  {strategy}: {info['total_samples']} samples ({info['n_real']} real, {info['n_synthetic']} synthetic)")
    
    print("\n✓ DataAugmenter test passed!")