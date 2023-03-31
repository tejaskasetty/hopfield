import numpy as np
import matplotlib.pyplot as plt
from small_world_hopfield_network import SmallWorldHopfieldNetwork

# Define experiment parameters
n = 100  # Number of neurons
sparsity_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]  # Sparsity levels to test
num_memories_list = [10, 20, 30, 40, 50]  # Number of memories to store
num_test_patterns = 100  # Number of test patterns to generate
num_trials = 5  # Number of trials to average over

# Generate random binary memories
memories = np.random.randint(2, size=(max(num_memories_list), n))

# Run experiment for each number of memories
for num_memories in num_memories_list:
    # Initialize arrays to store results
    mean_recall_rates = np.zeros(len(sparsity_levels))
    std_recall_rates = np.zeros(len(sparsity_levels))

    # Run experiment for each sparsity level
    for i, sparsity in enumerate(sparsity_levels):
        # Compute number of connections to add to achieve desired sparsity level
        k = int(sparsity * n / 2)

        # Initialize array to store recall rates for each trial
        recall_rates = np.zeros(num_trials)

        # Run experiment for each trial
        for trial in range(num_trials):
            # Generate random test patterns
            test_patterns = np.random.randint(2, size=(num_test_patterns, n))

            # Train network on random subset of memories
            indices = np.random.choice(num_memories, size=num_memories, replace=False)
            subset_memories = memories[indices]
            network = SmallWorldHopfieldNetwork(n)
            network.train(subset_memories)
            network.apply_small_world_topology(k=k, max_hops=2, clustering_prob=0.5)

            # Test network on random test patterns
            num_correct = 0
            for pattern in test_patterns:
                result = network.recall(pattern)
                if np.array_equal(result, pattern):
                    num_correct += 1
            recall_rates[trial] = num_correct / num_test_patterns

        # Compute mean and standard deviation of recall rates for this sparsity level
        mean_recall_rates[i] = np.mean(recall_rates)
        std_recall_rates[i] = np.std(recall_rates)

    # Plot results for this number of memories
    plt.errorbar(sparsity_levels, mean_recall_rates, yerr=std_recall_rates, label=f'{num_memories} memories')

# Add plot labels and legend
plt.xlabel('Sparsity level')
plt.ylabel('Recall rate')
plt.title('Small-world Hopfield network performance with varying sparsity and number of memories')
plt.legend()

# Show plot
plt.show()
