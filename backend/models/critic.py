"""
Critic network for WGAN-GP.
Distinguishes between real and synthetic gene expression profiles.
"""

import torch
import torch.nn as nn
from typing import List


class Critic(nn.Module):
    """
    Critic network (discriminator) for gene expression data.
    
    Architecture:
        Input: Gene expression profile ∈ R^num_features
        Hidden: Multiple fully connected layers with LeakyReLU and Dropout
        Output: Critic score (real-ness)
    
    Note: No BatchNorm for WGAN-GP critic (interferes with gradient penalty)
    """
    
    def __init__(
        self,
        num_features: int,
        hidden_dims: List[int] = [512, 256, 128],
        activation: str = "leaky_relu",
        leaky_slope: float = 0.2,
        dropout: float = 0.3,
        use_batch_norm: bool = False,
        spectral_norm: bool = False
    ):
        """
        Initialize Critic.
        
        Args:
            num_features: Dimension of input (number of genes)
            hidden_dims: List of hidden layer dimensions
            activation: Activation function (leaky_relu, relu, elu)
            leaky_slope: Negative slope for LeakyReLU
            dropout: Dropout rate
            use_batch_norm: Whether to use BatchNorm (not recommended for WGAN-GP)
            spectral_norm: Whether to use spectral normalization
        """
        super(Critic, self).__init__()
        
        self.num_features = num_features
        self.hidden_dims = hidden_dims
        
        # Build network
        layers = []
        input_dim = num_features
        
        # Hidden layers
        for hidden_dim in hidden_dims:
            # Linear layer
            linear = nn.Linear(input_dim, hidden_dim)
            
            # Apply spectral normalization if requested
            if spectral_norm:
                linear = nn.utils.spectral_norm(linear)
            
            layers.append(linear)
            
            # BatchNorm (not recommended for WGAN-GP)
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            
            # Activation
            if activation == "leaky_relu":
                layers.append(nn.LeakyReLU(leaky_slope))
            elif activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "elu":
                layers.append(nn.ELU())
            else:
                raise ValueError(f"Unknown activation: {activation}")
            
            # Dropout
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            
            input_dim = hidden_dim
        
        # Output layer (single value, no activation)
        output_linear = nn.Linear(input_dim, 1)
        if spectral_norm:
            output_linear = nn.utils.spectral_norm(output_linear)
        layers.append(output_linear)
        
        self.model = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor, return_intermediates: bool = False) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Gene expression profiles (batch_size × num_features)
            return_intermediates: Whether to return intermediate features for feature matching
            
        Returns:
            Critic scores (batch_size × 1), or tuple of (scores, intermediates) if return_intermediates=True
        """
        if return_intermediates:
            # Get intermediate features from the second-to-last layer
            intermediates = []
            current = x
            for layer in self.model[:-1]:  # All layers except the last
                current = layer(current)
                intermediates.append(current)
            final_output = self.model[-1](current)  # Last layer
            return final_output, intermediates
        else:
            return self.model(x)
    
    def get_intermediate_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get intermediate features for feature matching.
        
        Args:
            x: Input samples
            
        Returns:
            Intermediate features from the last hidden layer
        """
        _, intermediates = self.forward(x, return_intermediates=True)
        # Return features from the last intermediate layer (before final output)
        return intermediates[-1] if intermediates else x


class MinibatchDiscrimination(nn.Module):
    """
    Minibatch discrimination layer.
    Helps prevent mode collapse by allowing discriminator to compare samples.
    """
    
    def __init__(self, in_features: int, out_features: int, kernel_dims: int = 5):
        """
        Initialize minibatch discrimination.
        
        Args:
            in_features: Input dimension
            out_features: Output dimension
            kernel_dims: Number of kernels
        """
        super(MinibatchDiscrimination, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.kernel_dims = kernel_dims
        
        self.T = nn.Parameter(torch.randn(in_features, out_features, kernel_dims))
        nn.init.normal_(self.T, 0, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input (batch_size × in_features)
            
        Returns:
            Output (batch_size × (in_features + out_features))
        """
        # Compute minibatch features
        matrices = x.mm(self.T.view(self.in_features, -1))
        matrices = matrices.view(-1, self.out_features, self.kernel_dims)
        
        M = matrices.unsqueeze(0)  # 1 × batch_size × out_features × kernel_dims
        M_T = M.permute(1, 0, 2, 3)  # batch_size × 1 × out_features × kernel_dims
        
        # Compute L1 distance between samples
        norm = torch.abs(M - M_T).sum(3)  # batch_size × batch_size × out_features
        
        # Sum over batch dimension (excluding self)
        expnorm = torch.exp(-norm)
        o_b = (expnorm.sum(0) - 1)  # batch_size × out_features
        
        # Concatenate with input
        return torch.cat([x, o_b], 1)


class CriticWithMinibatch(nn.Module):
    """Critic with minibatch discrimination."""
    
    def __init__(
        self,
        num_features: int,
        hidden_dims: List[int] = [512, 256, 128],
        activation: str = "leaky_relu",
        leaky_slope: float = 0.2,
        dropout: float = 0.3,
        minibatch_kernel_dims: int = 5
    ):
        """Initialize critic with minibatch discrimination."""
        super(CriticWithMinibatch, self).__init__()
        
        # Initial layers
        layers = []
        input_dim = num_features
        
        for i, hidden_dim in enumerate(hidden_dims[:-1]):
            layers.append(nn.Linear(input_dim, hidden_dim))
            
            if activation == "leaky_relu":
                layers.append(nn.LeakyReLU(leaky_slope))
            
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            
            input_dim = hidden_dim
        
        self.features = nn.Sequential(*layers)
        
        # Minibatch discrimination
        self.minibatch = MinibatchDiscrimination(
            in_features=input_dim,
            out_features=hidden_dims[-1],
            kernel_dims=minibatch_kernel_dims
        )
        
        # Final layers
        final_dim = input_dim + hidden_dims[-1]
        self.final = nn.Sequential(
            nn.Linear(final_dim, hidden_dims[-1]),
            nn.LeakyReLU(leaky_slope),
            nn.Linear(hidden_dims[-1], 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.features(x)
        x = self.minibatch(x)
        x = self.final(x)
        return x


def compute_gradient_penalty(
    critic: nn.Module,
    real_samples: torch.Tensor,
    fake_samples: torch.Tensor,
    device: torch.device
) -> torch.Tensor:
    """
    Compute gradient penalty for WGAN-GP.
    
    Args:
        critic: Critic network
        real_samples: Real data samples
        fake_samples: Generated (fake) samples
        device: Device to compute on
        
    Returns:
        Gradient penalty term
    """
    batch_size = real_samples.size(0)
    
    # Random weight for interpolation
    alpha = torch.rand(batch_size, 1, device=device)
    
    # Interpolate between real and fake samples
    interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
    
    # Get critic scores for interpolated samples
    critic_interpolates = critic(interpolates)
    
    # Compute gradients
    gradients = torch.autograd.grad(
        outputs=critic_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(critic_interpolates),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    
    # Compute gradient penalty
    gradients = gradients.view(batch_size, -1)
    gradient_norm = gradients.norm(2, dim=1)
    gradient_penalty = ((gradient_norm - 1) ** 2).mean()
    
    return gradient_penalty


def test_critic():
    """Test critic network."""
    print("Testing Critic...")
    
    num_features = 2000
    batch_size = 32
    
    # Test standard critic
    critic = Critic(
        num_features=num_features,
        hidden_dims=[512, 256, 128]
    )
    
    print(f"\nCritic architecture:")
    print(critic)
    
    # Test forward pass
    x = torch.randn(batch_size, num_features)
    output = critic(x)
    
    print(f"\nForward pass:")
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Output range: [{output.min():.4f}, {output.max():.4f}]")
    
    # Count parameters
    n_params = sum(p.numel() for p in critic.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {n_params:,}")
    
    # Test gradient penalty
    fake_samples = torch.randn_like(x)
    gp = compute_gradient_penalty(critic, x, fake_samples, torch.device('cpu'))
    print(f"\nGradient penalty: {gp.item():.4f}")
    
    print("\n✓ Critic test passed!")


if __name__ == "__main__":
    test_critic()