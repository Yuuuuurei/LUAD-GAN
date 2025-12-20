"""
Loss functions for GAN training.
"""

import torch
import torch.nn as nn


def wasserstein_loss_generator(fake_scores: torch.Tensor) -> torch.Tensor:
    """
    Wasserstein loss for generator.
    Generator wants to maximize critic score on fake samples.
    
    Args:
        fake_scores: Critic scores for fake samples
        
    Returns:
        Generator loss
    """
    return -fake_scores.mean()


def wasserstein_loss_critic(
    real_scores: torch.Tensor,
    fake_scores: torch.Tensor
) -> torch.Tensor:
    """
    Wasserstein loss for critic.
    Critic wants to maximize difference between real and fake scores.
    
    Args:
        real_scores: Critic scores for real samples
        fake_scores: Critic scores for fake samples
        
    Returns:
        Negative Wasserstein distance (for minimization)
    """
    return fake_scores.mean() - real_scores.mean()


def vanilla_gan_loss_generator(fake_scores: torch.Tensor) -> torch.Tensor:
    """
    Vanilla GAN loss for generator.
    
    Args:
        fake_scores: Discriminator scores for fake samples
        
    Returns:
        Generator loss
    """
    criterion = nn.BCEWithLogitsLoss()
    target = torch.ones_like(fake_scores)
    return criterion(fake_scores, target)


def vanilla_gan_loss_discriminator(
    real_scores: torch.Tensor,
    fake_scores: torch.Tensor
) -> torch.Tensor:
    """
    Vanilla GAN loss for discriminator.
    
    Args:
        real_scores: Discriminator scores for real samples
        fake_scores: Discriminator scores for fake samples
        
    Returns:
        Discriminator loss
    """
    criterion = nn.BCEWithLogitsLoss()
    
    real_target = torch.ones_like(real_scores)
    fake_target = torch.zeros_like(fake_scores)
    
    real_loss = criterion(real_scores, real_target)
    fake_loss = criterion(fake_scores, fake_target)
    
    return real_loss + fake_loss


class WassersteinLoss:
    """Wasserstein loss wrapper."""
    
    def __init__(self):
        """Initialize Wasserstein loss."""
        pass
    
    def generator_loss(self, fake_scores: torch.Tensor) -> torch.Tensor:
        """Compute generator loss."""
        return wasserstein_loss_generator(fake_scores)
    
    def critic_loss(
        self,
        real_scores: torch.Tensor,
        fake_scores: torch.Tensor
    ) -> torch.Tensor:
        """Compute critic loss."""
        return wasserstein_loss_critic(real_scores, fake_scores)


class VanillaGANLoss:
    """Vanilla GAN loss wrapper."""
    
    def __init__(self):
        """Initialize vanilla GAN loss."""
        pass
    
    def generator_loss(self, fake_scores: torch.Tensor) -> torch.Tensor:
        """Compute generator loss."""
        return vanilla_gan_loss_generator(fake_scores)
    
    def discriminator_loss(
        self,
        real_scores: torch.Tensor,
        fake_scores: torch.Tensor
    ) -> torch.Tensor:
        """Compute discriminator loss."""
        return vanilla_gan_loss_discriminator(real_scores, fake_scores)


if __name__ == "__main__":
    # Test losses
    print("Testing loss functions...")
    
    batch_size = 32
    real_scores = torch.randn(batch_size, 1)
    fake_scores = torch.randn(batch_size, 1)
    
    # Test Wasserstein loss
    gen_loss = wasserstein_loss_generator(fake_scores)
    critic_loss = wasserstein_loss_critic(real_scores, fake_scores)
    
    print(f"\nWasserstein Loss:")
    print(f"  Generator loss: {gen_loss.item():.4f}")
    print(f"  Critic loss: {critic_loss.item():.4f}")
    
    # Test Vanilla GAN loss
    gen_loss_vanilla = vanilla_gan_loss_generator(fake_scores)
    disc_loss_vanilla = vanilla_gan_loss_discriminator(real_scores, fake_scores)
    
    print(f"\nVanilla GAN Loss:")
    print(f"  Generator loss: {gen_loss_vanilla.item():.4f}")
    print(f"  Discriminator loss: {disc_loss_vanilla.item():.4f}")
    
    print("\n✓ Loss functions test passed!")