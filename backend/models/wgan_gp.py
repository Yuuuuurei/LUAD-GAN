"""
WGAN-GP (Wasserstein GAN with Gradient Penalty) implementation.
Main module that combines Generator and Critic.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
from .generator import Generator
from .critic import Critic, compute_gradient_penalty


class WGAN_GP(nn.Module):
    """
    Wasserstein GAN with Gradient Penalty.
    
    Combines Generator and Critic networks for adversarial training.
    """
    
    def __init__(
        self,
        num_features: int,
        latent_dim: int = 128,
        generator_config: Optional[dict] = None,
        critic_config: Optional[dict] = None,
        gradient_penalty_weight: float = 10.0,
        variance_reg_config: Optional[dict] = None,
        feature_matching_config: Optional[dict] = None
    ):
        """
        Initialize WGAN-GP.
        
        Args:
            num_features: Dimension of data (number of genes)
            latent_dim: Dimension of latent vector z
            generator_config: Configuration for Generator
            critic_config: Configuration for Critic
            gradient_penalty_weight: Weight for gradient penalty (lambda)
            variance_reg_config: Configuration for variance regularization
            feature_matching_config: Configuration for feature matching
        """
        super(WGAN_GP, self).__init__()
        
        self.num_features = num_features
        self.latent_dim = latent_dim
        self.gradient_penalty_weight = gradient_penalty_weight
        self.variance_reg_config = variance_reg_config or {'enabled': False}
        self.feature_matching_config = feature_matching_config or {'enabled': False}
        self.feature_matching_config = feature_matching_config or {'enabled': False}
        
        # Default configurations
        if generator_config is None:
            generator_config = {
                'hidden_dims': [256, 512, 1024],
                'activation': 'leaky_relu',
                'use_batch_norm': True,
                'output_activation': 'tanh'
            }
        
        if critic_config is None:
            critic_config = {
                'hidden_dims': [512, 256, 128],
                'activation': 'leaky_relu',
                'dropout': 0.3,
                'use_batch_norm': False
            }
        
        # Initialize networks
        self.generator = Generator(
            latent_dim=latent_dim,
            num_features=num_features,
            **generator_config
        )
        
        self.critic = Critic(
            num_features=num_features,
            **critic_config
        )
    
    def generate(self, n_samples: int, device: Optional[torch.device] = None) -> torch.Tensor:
        """
        Generate synthetic samples.
        
        Args:
            n_samples: Number of samples to generate
            device: Device to generate on
            
        Returns:
            Generated samples (n_samples × num_features)
        """
        return self.generator.generate(n_samples, device)
    
    def compute_generator_loss(self, fake_samples: torch.Tensor, real_samples: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute generator loss (WGAN).
        
        Args:
            fake_samples: Generated samples
            real_samples: Real samples (optional, for variance regularization)
            
        Returns:
            Generator loss
        """
        # Generator wants to maximize critic score on fake samples
        # Equivalent to minimizing negative critic score
        fake_scores = self.critic(fake_samples)
        generator_loss = -fake_scores.mean()
        
        # Add regularization if enabled and real_samples provided
        if real_samples is not None:
            if hasattr(self, 'variance_reg_config') and self.variance_reg_config.get('enabled', False):
                var_loss = self._compute_variance_loss(fake_samples, real_samples)
                var_weight = self.variance_reg_config['weight']
                generator_loss = generator_loss + var_weight * var_loss
            elif hasattr(self, 'feature_matching_config') and self.feature_matching_config.get('enabled', False):
                fm_loss = self._compute_feature_matching_loss(fake_samples, real_samples)
                fm_weight = self.feature_matching_config['weight']
                generator_loss = generator_loss + fm_weight * fm_loss
        
        return generator_loss
    
    def _compute_variance_loss(self, fake_samples: torch.Tensor, real_samples: torch.Tensor) -> torch.Tensor:
        """
        Compute variance regularization loss.
        
        Args:
            fake_samples: Generated samples
            real_samples: Real samples
            
        Returns:
            Variance loss
        """
        import torch.nn.functional as F
        
        # Compute variance per feature
        real_var = torch.var(real_samples, dim=0, unbiased=False)
        fake_var = torch.var(fake_samples, dim=0, unbiased=False)
        
        # Target variance ratio
        target_ratio = self.variance_reg_config.get('target_ratio', 1.0)
        target_var = real_var * target_ratio
        
        # MSE loss between target and fake variances
        var_loss = F.mse_loss(fake_var, target_var)
        
        return var_loss
    
    def _compute_feature_matching_loss(self, fake_samples: torch.Tensor, real_samples: torch.Tensor) -> torch.Tensor:
        """
        Compute feature matching loss.
        Matches statistics of critic intermediate features between real and fake samples.
        
        Args:
            fake_samples: Generated samples
            real_samples: Real samples
            
        Returns:
            Feature matching loss
        """
        import torch.nn.functional as F
        
        # Get intermediate features from critic
        real_features = self.critic.get_intermediate_features(real_samples)
        fake_features = self.critic.get_intermediate_features(fake_samples)
        
        # Compute loss based on configured layers
        layers = self.feature_matching_config.get('layers', ['mean', 'std'])
        total_loss = 0
        
        for layer_name in layers:
            if layer_name == 'mean':
                real_stat = torch.mean(real_features, dim=0)
                fake_stat = torch.mean(fake_features, dim=0)
            elif layer_name == 'std':
                real_stat = torch.std(real_features, dim=0)
                fake_stat = torch.std(fake_features, dim=0)
            elif layer_name == 'var':
                real_stat = torch.var(real_features, dim=0)
                fake_stat = torch.var(fake_features, dim=0)
            
            # L2 loss between statistics
            layer_loss = F.mse_loss(fake_stat, real_stat)
            total_loss += layer_loss
        
        return total_loss
    
    def compute_critic_loss(
        self,
        real_samples: torch.Tensor,
        fake_samples: torch.Tensor,
        device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute critic loss (WGAN-GP).
        
        Args:
            real_samples: Real data samples
            fake_samples: Generated samples
            device: Device to compute on
            
        Returns:
            Tuple of (total loss, wasserstein distance, gradient penalty)
        """
        # Critic scores
        real_scores = self.critic(real_samples)
        fake_scores = self.critic(fake_samples)
        
        # Wasserstein distance (critic wants to maximize)
        wasserstein_distance = real_scores.mean() - fake_scores.mean()
        
        # Gradient penalty
        gradient_penalty = compute_gradient_penalty(
            self.critic,
            real_samples,
            fake_samples,
            device
        )
        
        # Total critic loss (minimize negative Wasserstein distance + gradient penalty)
        critic_loss = -wasserstein_distance + self.gradient_penalty_weight * gradient_penalty
        
        return critic_loss, wasserstein_distance, gradient_penalty
    
    def save(self, filepath: str):
        """
        Save model state.
        
        Args:
            filepath: Path to save model
        """
        torch.save({
            'generator_state_dict': self.generator.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'num_features': self.num_features,
            'latent_dim': self.latent_dim,
            'gradient_penalty_weight': self.gradient_penalty_weight
        }, filepath)
    
    def load(self, filepath: str, device: Optional[torch.device] = None):
        """
        Load model state.
        
        Args:
            filepath: Path to load model from
            device: Device to load to
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        checkpoint = torch.load(filepath, map_location=device)
        
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        
        self.num_features = checkpoint['num_features']
        self.latent_dim = checkpoint['latent_dim']
        self.gradient_penalty_weight = checkpoint['gradient_penalty_weight']
    
    def count_parameters(self) -> Tuple[int, int, int]:
        """
        Count parameters in generator and critic.
        
        Returns:
            Tuple of (total, generator, critic) parameter counts
        """
        gen_params = sum(p.numel() for p in self.generator.parameters() if p.requires_grad)
        critic_params = sum(p.numel() for p in self.critic.parameters() if p.requires_grad)
        total_params = gen_params + critic_params
        
        return total_params, gen_params, critic_params


def create_wgan_gp_from_config(config: dict, num_features: int) -> WGAN_GP:
    """
    Create WGAN-GP model from configuration.
    
    Args:
        config: Configuration dictionary
        num_features: Number of features in data
        
    Returns:
        Initialized WGAN-GP model
    """
    # Extract configurations
    arch_config = config['architecture']
    gen_config = arch_config['generator']
    critic_config = arch_config['critic']
    
    # Generator configuration
    generator_config = {
        'hidden_dims': gen_config['hidden_dims'],
        'activation': gen_config['activation'],
        'leaky_slope': gen_config['leaky_slope'],
        'use_batch_norm': gen_config['use_batch_norm'],
        'output_activation': gen_config['output_activation'],
        'dropout': gen_config['dropout']
    }
    
    # Critic configuration
    critic_cfg = {
        'hidden_dims': critic_config['hidden_dims'],
        'activation': critic_config['activation'],
        'leaky_slope': critic_config['leaky_slope'],
        'dropout': critic_config['dropout'],
        'use_batch_norm': critic_config['use_batch_norm'],
        'spectral_norm': critic_config['spectral_norm']
    }
    
    # Create model
    model = WGAN_GP(
        num_features=num_features,
        latent_dim=gen_config['latent_dim'],
        generator_config=generator_config,
        critic_config=critic_cfg,
        gradient_penalty_weight=config['training']['wgan_gp']['gradient_penalty_weight'],
        variance_reg_config=config['training']['regularization'].get('variance_regularization'),
        feature_matching_config=config['training']['regularization'].get('feature_matching')
    )
    
    return model


def test_wgan_gp():
    """Test WGAN-GP model."""
    print("Testing WGAN-GP...")
    
    num_features = 2000
    latent_dim = 128
    batch_size = 32
    device = torch.device('cpu')
    
    # Create model
    model = WGAN_GP(
        num_features=num_features,
        latent_dim=latent_dim
    )
    
    print(f"\nModel created:")
    total, gen, critic = model.count_parameters()
    print(f"  Total parameters: {total:,}")
    print(f"  Generator parameters: {gen:,}")
    print(f"  Critic parameters: {critic:,}")
    
    # Test generation
    print(f"\nTesting generation...")
    fake_samples = model.generate(n_samples=batch_size, device=device)
    print(f"  Generated shape: {fake_samples.shape}")
    print(f"  Generated range: [{fake_samples.min():.4f}, {fake_samples.max():.4f}]")
    
    # Test loss computation
    print(f"\nTesting loss computation...")
    real_samples = torch.randn(batch_size, num_features)
    
    # Generator loss
    gen_loss = model.compute_generator_loss(fake_samples)
    print(f"  Generator loss: {gen_loss.item():.4f}")
    
    # Critic loss
    critic_loss, wd, gp = model.compute_critic_loss(real_samples, fake_samples, device)
    print(f"  Critic loss: {critic_loss.item():.4f}")
    print(f"  Wasserstein distance: {wd.item():.4f}")
    print(f"  Gradient penalty: {gp.item():.4f}")
    
    # Test save/load
    print(f"\nTesting save/load...")
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
        model.save(f.name)
        print(f"  Saved to {f.name}")
        
        model2 = WGAN_GP(num_features=num_features, latent_dim=latent_dim)
        model2.load(f.name)
        print(f"  Loaded successfully")
    
    print("\n✓ WGAN-GP test passed!")


if __name__ == "__main__":
    test_wgan_gp()