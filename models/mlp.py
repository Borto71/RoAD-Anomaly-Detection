import torch
import torch.nn as nn

class TabularAnomalyMLP(nn.Module):
    """
    Autoencoder MLP for the detection of anomalies in tabular data.
    """
    def __init__(self, input_dim, hidden_dim = 64, bottleneck_dim = 32, p_drop = 0.2):
        """
        Args:
            input_dim:      Number of input features.
            hidden_dim:     Number of neurons in the hidden layers.
            bottleneck_dim: Number of neurons in the bottleneck layer.
            p_drop:         Dropout probability.
        """
        super().__init__()
        
            # ENCODER
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace = True),
            nn.Dropout(p_drop),
            
            nn.Linear(hidden_dim, bottleneck_dim) # Bottleneck layer (no activation)
        )
        
        # DECODER
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, hidden_dim),
            nn.ReLU(inplace = True),
            nn.Dropout(p_drop),
            
            nn.Linear(hidden_dim, input_dim) 
            # No final activation (regression task)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded