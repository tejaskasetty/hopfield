import torch
import numpy as np

class SmallWorldHopfieldNet(torch.nn.Module):
    def __init__(self, n_units, p, k, max_hops, clustering):
        super(SmallWorldHopfieldNet, self).__init__()
        self.n_units = n_units
        self.p = p
        self.k = k
        self.max_hops = max_hops
        self.clustering = clustering
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
        
        # Apply small-world topology to weight matrix
        self.apply_small_world_topology()
        
        # Normalize weight matrix by number of units
        self.W /= self.n_units
        
    def energy(self, y):
        # y: batch_size x n_units
        
        # Compute energy of states
        energy = -0.5 * torch.sum(torch.matmul(y, self.W) * y, axis=1)
        
        return energy
    
    def apply_small_world_topology(self):
        # Create regular lattice
        lattice = np.zeros((self.n_units, self.n_units))
        for i in range(self.n_units):
            for j in range(1, self.k+1):
                lattice[i, (i+j) % self.n_units] = 1
                lattice[i, (i-j) % self.n_units] = 1
                
        # Add random connections
        for i in range(self.n_units):
            for j in range(i+1, self.n_units):
                if np.random.binomial(1, self.p):
                    dist = min(abs(j-i), self.n_units - abs(j-i))
                    if dist <= self.max_hops and np.random.uniform() < self.clustering:
                        lattice[i, j] = 1
                        lattice[j, i] = 1
                    
        # Apply connections to weight matrix
        self.W *= torch.from_numpy(lattice).float()
    
    def recall(self, x):
        return self(x)
