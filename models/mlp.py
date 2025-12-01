import torch
import torch.nn as nn

class MLP(nn.Module): # classe MLP che estende nn.Module
    """
    Simple multilayer perceptron for Fashion-MNIST.
    Input: 28x28 grayscale images (flattened)
    Output: 10 classes
    """
    def __init__(self, in_dim = 28 * 28, num_classes = 10, p_drop = 0.2): # input from 28x28 images in grey scale, for 10 classes, dropout prob 0.2
        super().__init__()
        self.net = nn.Sequential( 
            nn.Flatten(), # we transform the 28x28 image into a flat vector
            nn.Linear(in_dim, 512), # connect the 784 point at the 512 neurons of the hidden layer
            nn.ReLU(inplace = True), # ReLu activation function, non-linear
            nn.Dropout(p_drop), # during training, disable randomly the 20% of the neurons to avoid overfitting
            
            nn.Linear(512, 256), # second layer, connect 512 to 256 neurons
            nn.ReLU(inplace = True), # ReLu activation function
            nn.Dropout(p_drop), # dropout for regularization
            
            nn.Linear(256, num_classes) # final layer, connect 256 neurons to 10 output classes
        )

    def forward(self, x):
        return self.net(x) # forward pass through the network