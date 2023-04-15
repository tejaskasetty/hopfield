import numpy as np
from random import randint
import matplotlib.pyplot as plt

def hamming_distance(x, y):
    return np.sum(x, y)

def sign(x):
    return np.where(x >= -1e-15, 1, -1)

def gen_partial_pattern(patterns):
    p, n = patterns.shape
    res = np.empty((p, n))
    for i, pattern in enumerate(patterns):
        is_zero_pattern = True
        while is_zero_pattern:
            mask = get_mask(n)
            res[i] = pattern * mask
            is_zero_pattern = np.prod(res[i] == 0)
    return res

def gen_random_pattern(p, n):
    #p: number of patterns
    #n: number of neurons
    return np.random.choice([-1, 1], size=(p, n))

def get_mask(n):
    val = [-1, 1]
    pos = randint(0, 1)
    mod = randint(1, n-1)
    tile = []
    for i in range(1, n+1):
        tile.append(val[pos])
        if i % mod == 0:
            pos = (pos + 1) % 2
    return np.array(tile)

def is_pattern_match(pat_1, pat_2):
    return np.array_equal(pat_1, pat_2) or np.array_equal(pat_1, -1 * pat_2)

def is_pattern_present(pattern, memories):
    n = memories.shape[1]
    x = np.absolute(pattern + memories).sum(axis = 1) / n
    d = np.unique(x)
    return np.isin(d, [0, 2]).sum() > 0

def cnt_patterns_present(patterns, memories):
    return sum([is_pattern_present(p, memories) for p in patterns])

def cnt_pattern_matches(pats_1, pats_2):
    return sum([ is_pattern_match(p_1, p_2) for (p_1, p_2) in zip(pats_1, pats_2)])


def plot_network_performance(num_memories_list, mean_recall_rates, std_recall_rates, label):
    # Plot results
    plt.errorbar(num_memories_list, mean_recall_rates, std_recall_rates, label = label)
    # Add plot labels
    plt.xlabel('Num of memories')
    plt.ylabel('Recall rate')
    plt.title(f'Hopfield network performance (recall {label})')
    plt.legend()