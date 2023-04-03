import numpy as np
from hopfield_network import HopfieldNetwork

# Generate a Hopfield network with 3 attractor states
patterns = np.array([[1, 1, -1, -1], [-1, -1, 1, 1], [1, -1, 1, -1]])
n_neurons = patterns.shape[1]
hopfield_net = HopfieldNetwork(n_neurons)
hopfield_net.train(patterns)

# Test synchronous updates
print('Testing synchronous updates...')
x_init = np.array([1, 1, 1, 1])
print(f'Initial state: {x_init}')
x_sync = hopfield_net.simulate(x_init, n_steps=1, update_method='sync')
print(f'Synchronous update: {x_sync}')

# Test asynchronous updates
print('\nTesting asynchronous updates...')
x_init = np.array([1, 1, 1, 1])
print(f'Initial state: {x_init}')
x_async = hopfield_net.simulate(x_init, n_steps=1, update_method='async')
print(f'Asynchronous update: {x_async}')