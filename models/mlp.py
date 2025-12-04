import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim = 128):

        """
        A Advanced Multi-Layer Perceptron (MLP) for classification tasks.
        Args:
            input_dim (int): Dimension of the input features.
            hidden_dim (int): Number of hidden units in the hidden layers.
        """
        super(MLP, self).__init__()
        # Block 1, big inital reduction
        self.layer1 = nn.Sequential(nn.Linear(input_dim, 512), nn.BatchNorm1d(512), nn.LeakyReLU(0.1), nn.Dropout(0.4))

        # Block 2
        self.layer2 = nn.Sequential(nn.Linear(512, 256), nn.BatchNorm1d(256), nn.LeakyReLU(0.1), nn.Dropout(0.3))

        # Block 3
        self.layer3 = nn.Sequential(nn.Linear(256, 128), nn.BatchNorm1d(128), nn.LeakyReLU(0.1), nn.Dropout(0.2))

        # Block 4
        self.layer4 = nn.Sequential(nn.Linear(128, 64), nn.BatchNorm1d(64), nn.LeakyReLU(0.1), nn.Dropout(0.1))

        # Block 5
        self.layer5 = nn.Sequential(nn.Linear(64, 32), nn.BatchNorm1d(32), nn.LeakyReLU(0.1), nn.Dropout(0.1))

        # Block 6
        self.layer6 = nn.Sequential(nn.Linear(32, 16), nn.BatchNorm1d(16), nn.LeakyReLU(0.1), nn.Dropout(0.1))

        # Block 7
        self.layer7 = nn.Sequential(nn.Linear(16, 8), nn.BatchNorm1d(8), nn.LeakyReLU(0.1), nn.Dropout(0.1))

        # Output layer
        self.fc_out = nn.Linear(8, 2)


    def forward(self, x):
        # Flattening [Batch, Channels, Time] -> [Batch, Channels*Time]
        x = x.reshape(x.size(0), -1)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)

        x = self.fc_out(x)
        return x