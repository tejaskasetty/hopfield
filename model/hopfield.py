import torch
import numpy as np

class HopfieldNet(torch.nn.Module):
    def __init__(self, n_units):
        super(HopfieldNet, self).__init__()
        self.n_units = n_units
        self.register_buffer('W', torch.zeros((n_units, n_units)))
        
    def forward(self, x):
        # x: batch_size x n_units
        
        # Compute activations
        h = torch.matmul(x, self.W)
        
        # Apply Heaviside step function to activations
        y = torch.sign(h)
        
        return y
    
    def train(self, x_train):
        # x_train: n_samples x n_units
        
        # Compute weight matrix using Hebbian learning rule
        self.W = torch.matmul(x_train.T, x_train)
        
        # Set diagonal elements of weight matrix to zero
        self.W.fill_diagonal_(0)
        
        # Normalize weight matrix by number of units
        self.W /= self.n_units
        
    def energy(self, y):
        # y: batch_size x n_units
        
        # Compute energy of states
        energy = -0.5 * torch.sum(torch.matmul(y, self.W) * y, axis=1)
        
        return energy
