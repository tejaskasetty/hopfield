import numpy as np
import matplotlib.pyplot as plt
from model import HopfieldNet, SmallWorldHopfieldNet
from utils import gen_partial_pattern, cnt_patterns_present, cnt_pattern_matches

# Test of the % of partial patterns that lead to any stored pattern recall
def test_for_any_valid_memory_recall(n_neurons = 100, num_trials = 20, num_test = 100):
    """
    n_neurons: number of neurons in the network
    num_trials: number of trials for each memory size
    num_test: default number of test patterns
    """
    num_memories_list = list(range(2, n_neurons // 2 + 1, 3))
    memories = np.random.choice([-1, 1], size = (max(num_memories_list), n_neurons)) # create random max memories
    mean_recall_rates = []
    std_recall_rates = []
    for i, num_memories in enumerate(num_memories_list):
        recall_rates = []
        #num_test = min(num_memories, dnum_test)
        for t in range(num_trials):
            indices = np.random.choice(num_memories, size = num_test)
            patterns = memories[indices, :]
            network = HopfieldNet(n_neurons)
            network.store(patterns)
            res = network.recall(gen_partial_pattern(patterns), update = 'async')
            num_correct = cnt_patterns_present(res, patterns)
            recall_rates.append(num_correct/num_test)
        mean_recall_rates.append(np.mean(recall_rates))
        std_recall_rates.append(np.std(recall_rates))
    
    return num_memories_list, mean_recall_rates, std_recall_rates

# Test the % of times partial patterns lead to correct pattern recall 
def test_for_exact_memory_recall(n_neurons = 100, num_trials = 20, num_test = 100):
    """
    n_neurons: number of neurons in the network
    num_trials: number of trials for each memory size
    num_test: default number of test patterns
    """
    num_memories_list = list(range(2, n_neurons // 2 + 1, 3))
    memories = np.random.choice([-1, 1], size = (max(num_memories_list), n_neurons)) # create random max memories
    mean_recall_rates = []
    std_recall_rates = []
    for i, num_memories in enumerate(num_memories_list):
        recall_rates = []
        #num_test = min(num_memories, dnum_test)
        for t in range(num_trials):
            indices = np.random.choice(num_memories, size = num_test)
            patterns = memories[indices, :]
            network = HopfieldNet(n_neurons)
            network.store(patterns)
            res = network.recall(gen_partial_pattern(patterns), update = 'async')
            num_correct = cnt_pattern_matches(res, patterns)
            recall_rates.append(num_correct/num_test)
        mean_recall_rates.append(np.mean(recall_rates))
        std_recall_rates.append(np.std(recall_rates))
    
    return num_memories_list, mean_recall_rates, std_recall_rates


def test_sw_hopfield_any_valid_pattern(n_neurons = 100, num_trials = 20, num_test = 100):
    """
    n_neurons: number of neurons in the network
    num_trials: number of trials for each memory size
    num_test: default number of test patterns
    """
    cluster_coeff = [0.2, 0.4, 0.6, 0.8, 1]  # Sparsity levels to test
    num_memories_list = list(range(2, n_neurons // 2 + 1, 3))
    memories = np.random.choice([-1, 1], size = (max(num_memories_list), n_neurons)) # create random memories

    
    # Run experiment for each clustering level
    for coeff in cluster_coeff:
        # Initialize arrays to store results
        mean_recall_rates = []
        std_recall_rates = []
        sparsity_rates = []
        # Run experiment for each number of memories
        for i, num_memories in enumerate(num_memories_list):
            # Initialize array to store recall rates for each trial
            recall_rates = []
            # Run experiment for each trial
            for t in range(num_trials):
                # Generate random test patterns
                indices = np.random.choice(num_memories, size = num_test)
                patterns = memories[indices, :]
                network = SmallWorldHopfieldNet(n_neurons, coeff, 0.1)
                sparsity_rates.append(network.sparsity())
                network.store(patterns)
                res = network.recall(gen_partial_pattern(patterns), update = 'async')
                num_correct = cnt_patterns_present(res, patterns)
                recall_rates.append(num_correct/num_test)

            # Compute mean and standard deviation of recall rates for this sparsity level
            mean_recall_rates.append(np.mean(recall_rates))
            std_recall_rates.append(np.std(recall_rates))

        print("Sparsity Rate: ", np.mean(sparsity_rates), coeff)
        # Plot results for this number of memories
        plt.errorbar(num_memories_list, mean_recall_rates, yerr=std_recall_rates, label=f'{coeff} clustering')
    # Add plot labels and legend
    plt.xlabel('Number of memories')
    plt.ylabel('Recall rate')
    plt.title('Hopfield network performance with varying sparsity and number of memories')
    plt.tight_layout()
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # Show plot
    plt.show()


def test_sw_hopfield_exact_pattern(n_neurons = 100, num_trials = 20, num_test = 100):
    """
    n_neurons: number of neurons in the network
    num_trials: number of trials for each memory size
    num_test: default number of test patterns
    """
    cluster_coeff = [0.2, 0.4, 0.6, 0.8, 1]  # Sparsity levels to test
    num_memories_list = list(range(2, n_neurons // 2 + 1, 3))
    memories = np.random.choice([-1, 1], size = (max(num_memories_list), n_neurons)) # create random memories

    
    # Run experiment for each clustering level
    for coeff in cluster_coeff:
        # Initialize arrays to store results
        mean_recall_rates = []
        std_recall_rates = []
        sparsity_rates = []
        # Run experiment for each number of memories
        for i, num_memories in enumerate(num_memories_list):
            # Initialize array to store recall rates for each trial
            recall_rates = []
            # Run experiment for each trial
            for t in range(num_trials):
                # Generate random test patterns
                indices = np.random.choice(num_memories, size = num_test)
                patterns = memories[indices, :]
                network = SmallWorldHopfieldNet(n_neurons, coeff, 0.1)
                sparsity_rates.append(network.sparsity())
                network.store(patterns)
                res = network.recall(gen_partial_pattern(patterns), update = 'async')
                num_correct = cnt_pattern_matches(res, patterns)
                recall_rates.append(num_correct/num_test)

            # Compute mean and standard deviation of recall rates for this sparsity level
            mean_recall_rates.append(np.mean(recall_rates))
            std_recall_rates.append(np.std(recall_rates))

        print("Sparsity Rate: ", np.mean(sparsity_rates), coeff)
        # Plot results for this number of memories
        plt.errorbar(num_memories_list, mean_recall_rates, yerr=std_recall_rates, label=f'{coeff} clustering')
    # Add plot labels and legend
    plt.xlabel('Number of memories')
    plt.ylabel('Recall rate')
    plt.title('Hopfield network performance with varying sparsity and number of memories')
    plt.tight_layout()
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # Show plot
    plt.show()