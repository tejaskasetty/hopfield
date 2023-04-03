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

# Update pattern at each step and store in array
for i in range(num_steps):
    pattern = network.recall(pattern)
    patterns[i + 1] = pattern

# Visualize patterns over time
fig, axs = plt.subplots(num_steps // 5 + 1, 5, figsize=(10, 10))
for i in range(num_steps + 1):
    ax = axs[i // 5, i % 5]
    ax.imshow(np.reshape(patterns[i], (10, 10)), cmap='binary')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f'Step {i}')
plt.show()
