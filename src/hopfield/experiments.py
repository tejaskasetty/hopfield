import numpy as np
import matplotlib.pyplot as plt
from model import HopfieldNet
from utils import gen_partial_pattern, cnt_patterns_present, cnt_pattern_matches

# Test of the % of partial patterns that lead to any stored pattern recall
def test_for_any_valid_memory_recall():
    n = 100
    num_memories_list = [3, 5, 7, 10, 13, 15, 25, 35, 50]
    num_test = 100
    memories = np.random.choice([-1, 1], size = (max(num_memories_list), n))
    num_trials = 20
    mean_recall_rates = []
    std_recall_rates = []
    for i, num_memories in enumerate(num_memories_list):
        recall_rates = []
        for t in range(num_trials):
            indices = np.random.choice(num_memories, size = num_test)
            patterns = memories[indices, :]
            network = HopfieldNet(n)
            network.store(patterns)
            res = network.recall(gen_partial_pattern(patterns), mode = 'async')
            num_correct = cnt_patterns_present(res, patterns)
            recall_rates.append(num_correct/num_test)
        mean_recall_rates.append(np.mean(recall_rates))
        std_recall_rates.append(np.std(recall_rates))

    # Plot results
    plt.errorbar(num_memories_list, mean_recall_rates, std_recall_rates, label = 'any stored pattern')
    # Add plot labels
    plt.xlabel('Num of memories')
    plt.ylabel('Recall rate')
    plt.title('Hopfield network performance (recall any stored pattern)')
    plt.legend()


# Test the % of times partial patterns lead to correct pattern recall 
def test_for_exact_memory_recall():
    n = 100
    num_memories_list = [3, 5, 7, 10, 13, 15, 25, 35, 50]
    num_test = 100
    memories = np.random.choice([-1, 1], size = (max(num_memories_list), n))
    network = HopfieldNet(n)
    network.store(memories)
    recall_rates = []
    num_trials = 20
    mean_recall_rates = []
    std_recall_rates = []
    for i, num_memories in enumerate(num_memories_list):
        recall_rates = []
        for t in range(num_trials):
            indices = np.random.choice(num_memories, size = num_test)
            patterns = memories[indices, :]
            network = HopfieldNet(n)
            network.store(patterns)
            res = network.recall(gen_partial_pattern(patterns), mode = 'async')
            num_correct = cnt_pattern_matches(res, patterns)
            recall_rates.append(num_correct/num_test)
        mean_recall_rates.append(np.mean(recall_rates))
        std_recall_rates.append(np.std(recall_rates))
    # Plot results
    plt.errorbar(num_memories_list, mean_recall_rates, std_recall_rates, label = 'correct_pattern')
    # Add plot labels
    plt.xlabel('Num of memories')
    plt.ylabel('Recall rate')
    plt.title('Hopfield network performance (recall correct pattern)')
    plt.legend()

def test_sparse_hopfield():
    # Define experiment parameters
    n = 100  # Number of neurons
    sparsity_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]  # Sparsity levels to test
    num_memories_list = [10, 20, 30, 40, 50]  # Number of memories to store
    num_test_patterns = 100  # Number of test patterns to generate
    num_trials = 5  # Number of trials to average over

    # Generate random binary memories
    memories = np.random.randint(2, size=(max(num_memories_list), n))
    print(memories)
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
                network = HopfieldNet(n)
                print(subset_memories)
                network.store(subset_memories)
                # network.apply_small_world_topology(k=k, max_hops=2, clustering_prob=0.5)

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
    plt.title('Hopfield network performance with varying sparsity and number of memories')
    plt.legend()

    # Show plot
    plt.show()