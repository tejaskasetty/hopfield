import numpy as np
import matplotlib.pyplot as plt
from model import HopfieldNetwork

# Define parameters
n = 100  # Number of neurons
num_memories = 5  # Number of memories to store
num_steps = 20  # Number of steps to run
noise_level = 0.2  # Level of random noise to add to initial pattern

# Generate random binary memories
memories = np.random.randint(2, size=(num_memories, n))

# Initialize network and train on memories
network = HopfieldNetwork(n)
network.train(memories)

# Generate initial pattern with added noise
pattern = memories[0]
noisy_pattern = pattern.copy()
noise = np.random.choice([-1, 1], size=n, p=[noise_level, 1 - noise_level])
noisy_pattern *= noise

# Initialize array to store patterns over time
patterns = np.zeros((num_steps + 1, n))
patterns[0] = noisy_pattern

# Initialize array to store energy over time
energies = np.zeros(num_steps + 1)
energies[0] = network.energy(noisy_pattern)

# Update pattern and energy at each step and store in arrays
for i in range(num_steps):
    pattern = network.recall(pattern)
    patterns[i + 1] = pattern
    energies[i + 1] = network.energy(pattern)

# Visualize energy landscape
fig, axs = plt.subplots(1, 2, figsize=(10, 4))
axs[0].plot(energies)
axs[0].set_xlabel('Step')
axs[0].set_ylabel('Energy')
axs[0].set_title('Energy Landscape')

# Visualize attractor dynamics
axs[1].imshow(np.reshape(patterns[0], (10, 10)), cmap='binary')
axs[1].set_xticks([])
axs[1].set_yticks([])
axs[1].set_title('Initial Pattern')
for i in range(1, num_steps + 1):
    axs[1].imshow(np.reshape(patterns[i], (10, 10)), cmap='binary')
    axs[1].set_title(f'Step {i}')
    plt.pause(0.5)

plt.show()
