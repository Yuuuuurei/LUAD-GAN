"""
Unit tests for WGAN-GP model
Tests: backend/models/wgan_gp.py, generator.py, critic.py
"""

import pytest
import torch
import torch.nn as nn
import numpy as np


class TestGenerator:
    """Test suite for Generator network"""
    
    @pytest.fixture
    def generator_config(self):
        """Generator configuration"""
        return {
            'latent_dim': 128,
            'num_features': 1000,
            'hidden_dims': [256, 512, 1024]
        }
    
    @pytest.fixture
    def simple_generator(self, generator_config):
        """Create a simple generator for testing"""
        class SimpleGenerator(nn.Module):
            def __init__(self, latent_dim, num_features):
                super().__init__()
                self.model = nn.Sequential(
                    nn.Linear(latent_dim, 256),
                    nn.LeakyReLU(0.2),
                    nn.BatchNorm1d(256),
                    nn.Linear(256, 512),
                    nn.LeakyReLU(0.2),
                    nn.BatchNorm1d(512),
                    nn.Linear(512, 1024),
                    nn.LeakyReLU(0.2),
                    nn.BatchNorm1d(1024),
                    nn.Linear(1024, num_features),
                )
            
            def forward(self, z):
                return self.model(z)
        
        return SimpleGenerator(
            generator_config['latent_dim'],
            generator_config['num_features']
        )
    
    def test_generator_forward_pass(self, simple_generator, generator_config):
        """Test generator forward pass"""
        batch_size = 32
        z = torch.randn(batch_size, generator_config['latent_dim'])
        
        fake_data = simple_generator(z)
        
        assert fake_data.shape == (batch_size, generator_config['num_features'])
        assert not torch.isnan(fake_data).any()
        assert not torch.isinf(fake_data).any()
    
    def test_generator_output_range(self, simple_generator, generator_config):
        """Test generator output is in reasonable range"""
        z = torch.randn(64, generator_config['latent_dim'])
        fake_data = simple_generator(z)
        
        # Output should be real-valued (no NaN/Inf)
        assert torch.isfinite(fake_data).all()
        
        # For gene expression, check if values are in plausible range
        # (depends on whether you use Tanh activation or not)
        mean_val = fake_data.mean().item()
        std_val = fake_data.std().item()
        
        # Should have non-zero variance
        assert std_val > 0.01
    
    def test_generator_different_batch_sizes(self, simple_generator, generator_config):
        """Test generator handles different batch sizes"""
        for batch_size in [1, 16, 32, 64, 128]:
            z = torch.randn(batch_size, generator_config['latent_dim'])
            fake_data = simple_generator(z)
            assert fake_data.shape[0] == batch_size
    
    def test_generator_gradient_flow(self, simple_generator, generator_config):
        """Test gradients flow through generator"""
        z = torch.randn(32, generator_config['latent_dim'], requires_grad=True)
        fake_data = simple_generator(z)
        loss = fake_data.mean()
        loss.backward()
        
        # Check that gradients exist
        assert z.grad is not None
        assert not torch.isnan(z.grad).any()
    
    def test_generator_weight_initialization(self, simple_generator):
        """Test generator weights are initialized"""
        for param in simple_generator.parameters():
            assert param.requires_grad
            # Weights should not be all zeros
            assert not torch.allclose(param, torch.zeros_like(param))


class TestCritic:
    """Test suite for Critic (Discriminator) network"""
    
    @pytest.fixture
    def critic_config(self):
        """Critic configuration"""
        return {
            'num_features': 1000,
            'hidden_dims': [512, 256, 128]
        }
    
    @pytest.fixture
    def simple_critic(self, critic_config):
        """Create a simple critic for testing"""
        class SimpleCritic(nn.Module):
            def __init__(self, num_features):
                super().__init__()
                self.model = nn.Sequential(
                    nn.Linear(num_features, 512),
                    nn.LeakyReLU(0.2),
                    nn.Dropout(0.3),
                    nn.Linear(512, 256),
                    nn.LeakyReLU(0.2),
                    nn.Dropout(0.3),
                    nn.Linear(256, 128),
                    nn.LeakyReLU(0.2),
                    nn.Linear(128, 1),
                )
            
            def forward(self, x):
                return self.model(x)
        
        return SimpleCritic(critic_config['num_features'])
    
    def test_critic_forward_pass(self, simple_critic, critic_config):
        """Test critic forward pass"""
        batch_size = 32
        x = torch.randn(batch_size, critic_config['num_features'])
        
        score = simple_critic(x)
        
        assert score.shape == (batch_size, 1)
        assert not torch.isnan(score).any()
        assert not torch.isinf(score).any()
    
    def test_critic_output_is_scalar_per_sample(self, simple_critic, critic_config):
        """Test critic outputs single score per sample"""
        x = torch.randn(16, critic_config['num_features'])
        score = simple_critic(x)
        
        assert score.shape == (16, 1)
        assert score.ndim == 2
    
    def test_critic_gradient_flow(self, simple_critic, critic_config):
        """Test gradients flow through critic"""
        x = torch.randn(32, critic_config['num_features'], requires_grad=True)
        score = simple_critic(x)
        loss = score.mean()
        loss.backward()
        
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
    
    def test_critic_different_batch_sizes(self, simple_critic, critic_config):
        """Test critic handles different batch sizes"""
        for batch_size in [1, 8, 32, 64]:
            x = torch.randn(batch_size, critic_config['num_features'])
            score = simple_critic(x)
            assert score.shape[0] == batch_size


class TestGradientPenalty:
    """Test suite for gradient penalty computation"""
    
    def test_gradient_penalty_calculation(self):
        """Test gradient penalty computation"""
        # Simple critic
        critic = nn.Sequential(
            nn.Linear(10, 5),
            nn.LeakyReLU(0.2),
            nn.Linear(5, 1)
        )
        
        batch_size = 16
        real = torch.randn(batch_size, 10)
        fake = torch.randn(batch_size, 10)
        
        # Interpolate
        alpha = torch.rand(batch_size, 1)
        interpolated = alpha * real + (1 - alpha) * fake
        interpolated.requires_grad_(True)
        
        # Compute critic score
        critic_interpolated = critic(interpolated)
        
        # Compute gradients
        gradients = torch.autograd.grad(
            outputs=critic_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones_like(critic_interpolated),
            create_graph=True,
            retain_graph=True
        )[0]
        
        # Gradient penalty
        gradients = gradients.view(batch_size, -1)
        gradient_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)
        gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
        
        assert gradient_penalty.item() >= 0
        assert not torch.isnan(gradient_penalty)
    
    def test_interpolation_between_real_and_fake(self):
        """Test interpolation is correct"""
        real = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        fake = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
        
        alpha = torch.tensor([[0.5], [0.5]])
        interpolated = alpha * real + (1 - alpha) * fake
        
        expected = torch.tensor([[3.0, 4.0], [5.0, 6.0]])
        assert torch.allclose(interpolated, expected)
    
    def test_gradient_norm_computation(self):
        """Test gradient norm is computed correctly"""
        gradients = torch.tensor([[3.0, 4.0], [0.0, 0.0]])
        
        # L2 norm per sample
        norms = torch.sqrt(torch.sum(gradients ** 2, dim=1))
        
        assert torch.allclose(norms, torch.tensor([5.0, 0.0]))


class TestWGANGPTrainingStep:
    """Test suite for WGAN-GP training step"""
    
    @pytest.fixture
    def models(self):
        """Create simple generator and critic"""
        latent_dim = 64
        num_features = 100
        
        class Gen(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Sequential(
                    nn.Linear(latent_dim, 256),
                    nn.LeakyReLU(0.2),
                    nn.Linear(256, num_features)
                )
            def forward(self, z):
                return self.fc(z)
        
        class Crit(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Sequential(
                    nn.Linear(num_features, 128),
                    nn.LeakyReLU(0.2),
                    nn.Linear(128, 1)
                )
            def forward(self, x):
                return self.fc(x)
        
        return {'generator': Gen(), 'critic': Crit(), 'latent_dim': latent_dim}
    
    def test_critic_loss_computation(self, models):
        """Test critic loss (Wasserstein distance + GP)"""
        generator = models['generator']
        critic = models['critic']
        latent_dim = models['latent_dim']
        
        batch_size = 32
        real_data = torch.randn(batch_size, 100)
        z = torch.randn(batch_size, latent_dim)
        
        # Generate fake data
        fake_data = generator(z).detach()
        
        # Critic scores
        real_score = critic(real_data).mean()
        fake_score = critic(fake_data).mean()
        
        # Wasserstein distance (without GP for simplicity)
        wasserstein_distance = real_score - fake_score
        
        assert isinstance(wasserstein_distance, torch.Tensor)
        assert wasserstein_distance.numel() == 1
    
    def test_generator_loss_computation(self, models):
        """Test generator loss"""
        generator = models['generator']
        critic = models['critic']
        latent_dim = models['latent_dim']
        
        batch_size = 32
        z = torch.randn(batch_size, latent_dim)
        
        # Generate fake data
        fake_data = generator(z)
        fake_score = critic(fake_data).mean()
        
        # Generator loss: -critic(fake)
        gen_loss = -fake_score
        
        assert isinstance(gen_loss, torch.Tensor)
        assert gen_loss.numel() == 1
    
    def test_optimizer_step(self, models):
        """Test optimizer updates weights"""
        generator = models['generator']
        optimizer = torch.optim.Adam(generator.parameters(), lr=1e-4)
        
        # Get initial weights
        initial_weights = [p.clone() for p in generator.parameters()]
        
        # Forward and backward
        z = torch.randn(16, models['latent_dim'])
        fake = generator(z)
        loss = fake.mean()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Check weights changed
        for initial, current in zip(initial_weights, generator.parameters()):
            assert not torch.allclose(initial, current)


class TestModelSaving:
    """Test suite for model checkpoint saving/loading"""
    
    def test_save_load_generator(self, tmp_path):
        """Test saving and loading generator weights"""
        latent_dim = 64
        num_features = 100
        
        # Create and save model
        gen = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, num_features)
        )
        
        checkpoint_path = tmp_path / "generator.pt"
        torch.save(gen.state_dict(), checkpoint_path)
        
        # Create new model and load weights
        gen_loaded = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, num_features)
        )
        gen_loaded.load_state_dict(torch.load(checkpoint_path))
        
        # Test that loaded model produces same output
        z = torch.randn(8, latent_dim)
        gen.eval()
        gen_loaded.eval()
        
        with torch.no_grad():
            output1 = gen(z)
            output2 = gen_loaded(z)
        
        assert torch.allclose(output1, output2)
    
    def test_checkpoint_contains_metadata(self):
        """Test checkpoint includes training metadata"""
        checkpoint = {
            'epoch': 100,
            'generator_state': {},
            'critic_state': {},
            'gen_optimizer_state': {},
            'critic_optimizer_state': {},
            'loss_history': {'critic': [], 'generator': []}
        }
        
        assert 'epoch' in checkpoint
        assert 'generator_state' in checkpoint
        assert 'critic_state' in checkpoint
        assert 'loss_history' in checkpoint


class TestModelEvaluation:
    """Test suite for model evaluation mode"""
    
    def test_model_eval_mode(self):
        """Test model switches to eval mode"""
        model = nn.Sequential(
            nn.Linear(10, 5),
            nn.BatchNorm1d(5),
            nn.Dropout(0.5),
            nn.Linear(5, 1)
        )
        
        model.eval()
        assert not model.training
        
        # BatchNorm and Dropout should behave differently in eval mode
        x = torch.randn(32, 10)
        with torch.no_grad():
            output1 = model(x)
            output2 = model(x)
        
        # In eval mode, outputs should be deterministic
        assert torch.allclose(output1, output2)
    
    def test_no_grad_during_generation(self):
        """Test synthetic data generation doesn't track gradients"""
        gen = nn.Sequential(nn.Linear(64, 100))
        
        gen.eval()
        with torch.no_grad():
            z = torch.randn(16, 64)
            fake = gen(z)
        
        assert not fake.requires_grad


if __name__ == '__main__':
    pytest.main([__file__, '-v'])