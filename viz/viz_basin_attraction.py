import numpy as np
import matplotlib.pyplot as plt
from hopfield_network import HopfieldNetwork

# Generate a Hopfield network with 3 attractor states
patterns = np.array([[1, 1, -1, -1], [-1, -1, 1, 1], [1, -1, 1, -1]])
n_neurons = patterns.shape[1]
hopfield_net = HopfieldNetwork(n_neurons)
hopfield_net.train(patterns)

# Define the energy landscape
resolution = 0.1
x = np.arange(-1.5, 1.5, resolution)
y = np.arange(-1.5, 1.5, resolution)
X, Y = np.meshgrid(x, y)
energy_landscape = np.zeros((len(x), len(y)))
for i in range(len(x)):
    for j in range(len(y)):
        energy_landscape[i, j] = hopfield_net.energy(np.array([x[i], y[j], x[i], y[j]]))

# Visualize the energy landscape
plt.figure()
plt.contourf(X, Y, energy_landscape.T, cmap=plt.cm.jet)
plt.colorbar()
plt.title('Energy Landscape')
plt.xlabel('x1')
plt.ylabel('x2')

# Simulate the state trajectory starting from a random initial state
n_steps = 100
x_init = np.random.rand(n_neurons) * 2 - 1
x_traj = hopfield_net.simulate(x_init, n_steps)

# Visualize the state trajectory and the basins of attraction
plt.figure()
for i in range(n_steps):
    state = x_traj[i]
    energy = hopfield_net.energy(state)
    attractor = np.argmin(np.sum((patterns - state)**2, axis=1))
    color = ['r', 'g', 'b'][attractor]
    plt.plot(state[0], state[1], color=color, marker='o', markersize=3)
plt.title('Basins of Attraction')
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()
