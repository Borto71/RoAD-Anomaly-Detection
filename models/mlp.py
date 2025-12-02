import torch
import torch.nn as nn
from typing import Tuple

class WindowedMLP(nn.Module):
    """
    Autoencoder MLP adapted for multi-variate time series windows.
    
    This model mimics the interface of the 1D CNN but uses fully connected layers.
    It flattens the input window (B, C, T) into a vector, encodes it, and 
    attempts to reconstruct the original window.
    
    Unsupervised Learning Approach:
    - Reconstruction-based: Train to minimize MSE(x, output).
    - Embedding-based: Use forward_features() to get the bottleneck 'z'.
    """

    def __init__(
        self, 
        in_channels: int, 
        window_size: int, 
        hidden_dim: int = 128, 
        bottleneck_dim: int = 2, 
        dropout: float = 0.2
    ) -> None:
        """
        Args:
            in_channels:    Number of input features (C).
            window_size:    Temporal length of the window (T).
            hidden_dim:     Number of neurons in the hidden layers.
            bottleneck_dim: Dimension of the latent space (z).
            dropout:        Dropout probability.
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.window_size = window_size
        
        # Calculate the flattened input dimension: C * T
        self.flat_dim = in_channels * window_size
        
        # ENCODER
        # Compresses the flattened time window into a latent vector.
        self.encoder = nn.Sequential(
            nn.Linear(self.flat_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),  # BatchNorm helps with training stability
            nn.GELU(),                   # GELU is often preferred over ReLU in modern nets
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, bottleneck_dim) 
            # Note: No activation on the bottleneck (z)
        )
        
        # DECODER 
        # Reconstructs the flattened time window from the latent vector.
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, self.flat_dim) 
            # No final activation (assuming input data is normalized, e.g., Z-score)
        )

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extracts the latent embedding for the input window.
        Useful for methods like Deep SVDD or clustering.
        
        Args:
            x: Tensor of shape (B, C, T)
            
        Returns:
            z: Tensor of shape (B, bottleneck_dim)
        """
        B, C, T = x.shape
        
        # Flatten: (B, C, T) -> (B, C * T)
        x_flat = x.reshape(B, -1)
        
        # Encode to latent space
        z = self.encoder(x_flat)
        return z

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Standard Autoencoder forward pass (Reconstruction).
        
        Args:
            x: Tensor of shape (B, C, T)
            
        Returns:
            x_hat: Reconstructed tensor of shape (B, C, T)
        """
        B, C, T = x.shape
        
        # 1. Flatten input
        x_flat = x.reshape(B, -1)
        
        # 2. Encode
        z = self.encoder(x_flat)
        
        # 3. Decode
        x_rec_flat = self.decoder(z)
        
        # 4. Reshape back to original temporal format (B, C, T)
        x_rec = x_rec_flat.reshape(B, C, T)
        
        return x_rec

def build_mlp_model(
    in_channels: int, 
    window_size: int, 
    hidden_dim: int = 128, 
    bottleneck_dim: int = 2,
    dropout: float = 0.2
) -> WindowedMLP:
    """
    Factory function to create the MLP model, similar to the CNN's build_model.
    """
    return WindowedMLP(
        in_channels = in_channels,
        window_size = window_size,
        hidden_dim = hidden_dim,
        bottleneck_dim = bottleneck_dim,
        dropout = dropout
    )