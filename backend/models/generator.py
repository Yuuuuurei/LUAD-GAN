"""
Generator network for WGAN-GP.
Generates synthetic gene expression profiles from latent vectors.
"""

import torch
import torch.nn as nn
from typing import List, Optional


class Generator(nn.Module):
    """
    Generator network for gene expression data.
    
    Architecture:
        Input: Latent vector z ∈ R^latent_dim
        Hidden: Multiple fully connected layers with BatchNorm and LeakyReLU
        Output: Synthetic gene expression profile ∈ R^num_features
    """
    
    def __init__(
        self,
        latent_dim: int,
        num_features: int,
        hidden_dims: List[int] = [256, 512, 1024],
        activation: str = "leaky_relu",
        leaky_slope: float = 0.2,
        use_batch_norm: bool = True,
        output_activation: str = "tanh",
        dropout: float = 0.0
    ):
        """
        Initialize Generator.
        
        Args:
            latent_dim: Dimension of latent vector z
            num_features: Dimension of output (number of genes)
            hidden_dims: List of hidden layer dimensions
            activation: Activation function (leaky_relu, relu, elu)
            leaky_slope: Negative slope for LeakyReLU
            use_batch_norm: Whether to use BatchNormalization
            output_activation: Output activation (tanh, linear, sigmoid)
            dropout: Dropout rate (0 = no dropout)
        """
        super(Generator, self).__init__()
        
        self.latent_dim = latent_dim
        self.num_features = num_features
        self.hidden_dims = hidden_dims
        
        # Build network
        layers = []
        input_dim = latent_dim
        
        # Hidden layers
        for hidden_dim in hidden_dims:
            # Linear layer
            layers.append(nn.Linear(input_dim, hidden_dim))
            
            # BatchNorm
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
        
        # Output layer
        layers.append(nn.Linear(input_dim, num_features))
        
        # Output activation
        if output_activation == "tanh":
            layers.append(nn.Tanh())
        elif output_activation == "sigmoid":
            layers.append(nn.Sigmoid())
        elif output_activation == "linear":
            pass  # No activation
        else:
            raise ValueError(f"Unknown output activation: {output_activation}")
        
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
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            z: Latent vectors (batch_size × latent_dim)
            
        Returns:
            Generated samples (batch_size × num_features)
        """
        return self.model(z)
    
    def generate(self, n_samples: int, device: Optional[torch.device] = None) -> torch.Tensor:
        """
        Generate synthetic samples.
        
        Args:
            n_samples: Number of samples to generate
            device: Device to generate on
            
        Returns:
            Generated samples (n_samples × num_features)
        """
        if device is None:
            device = next(self.parameters()).device
        
        # Sample latent vectors
        z = torch.randn(n_samples, self.latent_dim, device=device)
        
        # Generate samples
        with torch.no_grad():
            samples = self.forward(z)
        
        return samples


class ResidualGenerator(nn.Module):
    """
    Generator with residual connections.
    More stable for very deep networks.
    """
    
    def __init__(
        self,
        latent_dim: int,
        num_features: int,
        hidden_dims: List[int] = [256, 512, 1024],
        activation: str = "leaky_relu",
        leaky_slope: float = 0.2,
        use_batch_norm: bool = True,
        output_activation: str = "tanh"
    ):
        """Initialize Residual Generator."""
        super(ResidualGenerator, self).__init__()
        
        self.latent_dim = latent_dim
        self.num_features = num_features
        
        # Input projection
        self.input_proj = nn.Linear(latent_dim, hidden_dims[0])
        
        # Residual blocks
        self.blocks = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            self.blocks.append(
                ResidualBlock(
                    hidden_dims[i],
                    hidden_dims[i+1],
                    activation=activation,
                    leaky_slope=leaky_slope,
                    use_batch_norm=use_batch_norm
                )
            )
        
        # Output layer
        self.output = nn.Linear(hidden_dims[-1], num_features)
        
        # Output activation
        if output_activation == "tanh":
            self.output_activation = nn.Tanh()
        elif output_activation == "sigmoid":
            self.output_activation = nn.Sigmoid()
        else:
            self.output_activation = nn.Identity()
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.input_proj(z)
        
        for block in self.blocks:
            x = block(x)
        
        x = self.output(x)
        x = self.output_activation(x)
        
        return x


class ResidualBlock(nn.Module):
    """Residual block for generator."""
    
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        activation: str = "leaky_relu",
        leaky_slope: float = 0.2,
        use_batch_norm: bool = True
    ):
        """Initialize residual block."""
        super(ResidualBlock, self).__init__()
        
        layers = [nn.Linear(in_dim, out_dim)]
        
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(out_dim))
        
        if activation == "leaky_relu":
            layers.append(nn.LeakyReLU(leaky_slope))
        elif activation == "relu":
            layers.append(nn.ReLU())
        
        self.block = nn.Sequential(*layers)
        
        # Skip connection
        if in_dim != out_dim:
            self.skip = nn.Linear(in_dim, out_dim)
        else:
            self.skip = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection."""
        return self.block(x) + self.skip(x)


def test_generator():
    """Test generator networks."""
    print("Testing Generator...")
    
    latent_dim = 128
    num_features = 2000
    batch_size = 32
    
    # Test standard generator
    generator = Generator(
        latent_dim=latent_dim,
        num_features=num_features,
        hidden_dims=[256, 512, 1024]
    )
    
    print(f"\nGenerator architecture:")
    print(generator)
    
    # Test forward pass
    z = torch.randn(batch_size, latent_dim)
    output = generator(z)
    
    print(f"\nForward pass:")
    print(f"  Input shape: {z.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Output range: [{output.min():.4f}, {output.max():.4f}]")
    
    # Test generation
    samples = generator.generate(n_samples=10)
    print(f"\nGenerate 10 samples:")
    print(f"  Shape: {samples.shape}")
    
    # Count parameters
    n_params = sum(p.numel() for p in generator.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {n_params:,}")
    
    print("\n✓ Generator test passed!")


if __name__ == "__main__":
    test_generator()