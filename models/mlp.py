import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(MLP, self).__init__()

        # first layer, input dimension to 128 hidden units
        self.fc1 = nn.Linear(input_dim, 128)

        # Dropout layer for regularization
        self.dropout1 = nn.Dropout(p = 0.3)

        # second layer, 128 hidden units to output dimension
        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(0.3)

        # Output layer 
        self.fc3 = nn.Linear(64, 2)

        self.relu = nn.ReLU()

    def forward(self, x):
        x.view(x.size(0), -1)

        # Through first layer
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout1(x)

        # Secondo layer
        x = self.fc2(x)
        x = self.dropout2(x)

        # Output

        x = self.fc3

        return x