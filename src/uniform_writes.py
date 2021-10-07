import math
import numpy as np
import random

def construct_consecutive_changes_dict(data):
    """
    Calculates the number of blocks changes and tabulates the number of n blocks changed.

    data: raw change data, newline separated, where 1 is a change and 0 is no change
    """
    consecutive_changes_dict = {}
    consecutive_changes = 0

    for i in data:
        if i == 1:
            consecutive_changes += 1
        elif consecutive_changes != 0:
            if not consecutive_changes in consecutive_changes_dict:
                consecutive_changes_dict[consecutive_changes] = 1
            else:
                consecutive_changes_dict[consecutive_changes] += 1

            consecutive_changes = 0

    # This is needed to capture chains that end because the change record ends.
    if consecutive_changes != 0:
        if not consecutive_changes in consecutive_changes_dict:
            consecutive_changes_dict[consecutive_changes] = 1
        else:
            consecutive_changes_dict[consecutive_changes] += 1

    return consecutive_changes_dict

def consecutive_change_dict_to_matrix(dict):
    """
    Consumes a dictionary of keys, values where the keys are lengths of chains and the values are the number of occurences.
    It produces this in a numpy matrix form, where the first column is the chain length, the second is the number of occurences, and the third is the probability of a chain of length n.

    dict: the consecutive change dictionary
    """
    total_chains = np.sum(list(dict.values()))
    max_chain = np.max([int(x) for x in list(dict.keys())]) #

    matrix = np.append(np.array(list(range(1, max_chain + 1))).reshape(-1, 1), np.zeros((max_chain, 2)), axis=1) # 3

    for (key, value) in dict.items():
        key = int(key)
        matrix[key - 1, 1] = value
        matrix[key - 1, 2] = value / total_chains

    return matrix

def comb(n, r):
    """
    Returns the number of combinations of n elements where you choose r.

    n: the number of elements
    r: the elements you are choosing
    """
    if r <= n:
        return math.factorial(n) / (math.factorial(r) * math.factorial(n - r))
    else:
        return 0

def chains_per_partition(part):
    """
    Counts the number of occurences of a chain in a particular partition.

    part: a list of numbers representing a partition of a number
    """
    r = np.zeros(sum(part))
    for p in part:
        r[p - 1] += 1

    return r

def partition(number):
    """
    Produces all possible partitions of the number.

    number: the number
    """
    answer = set()
    answer.add((number,))
    for x in range(1, number):
        for y in partition(number - x):
            answer.add(tuple((x, ) + y))
    return answer

def all_partitions(n):
    """
    Returns all partitions of n in list form.

    n: the number to partition
    """
    return [list(x) for x in partition(n)]

def chain_probability(n, k):
    """
    Calculates the probability of chains of length <=k given n, when k writes are made uniformly over n blocks.

    n: the number of writeable blocks
    k: the number of uniform writes
    """
    prob_per_chain = np.zeros(k)
    expected_chains = 0
    expected_chains_by_size = np.zeros(k)
    for p in all_partitions(k):
        partition_probability = comb(n - k + 1, len(p)) / comb(n, k)

        chains = chains_per_partition(p)

        prob_in_chains = chains / np.sum(chains)
        prob_per_chain += prob_in_chains * partition_probability

        expected_chains += partition_probability * len(p)

        expected_chains_by_size += partition_probability * chains

    return prob_per_chain, expected_chains, expected_chains_by_size

def random_writes(disk_size, writes):
    """
    Makes writes distict writes to an array of disk_size size at random.

    disk_size: the number of writable blocks
    writes: the number of uniform writes
    """
    disk = np.zeros(disk_size)
    for i in range(writes):
        i = random.randint(0, disk_size - 1)
        while disk[i] == 1:
            i = random.randint(0, disk_size - 1)
        disk[i] = 1

    return disk

def experimental_proportion_of_singletons_per_write(disk_size, writes, samples=1):
    """
    Computes the probability of a singleton (a chain of size 1) experimentally for a disk of size disk_size and writes number of writes.

    disk_size: the number of writable blocks
    writes: the number of uniform writes
    samples: the number of samples to average over (for large disk_size and writes, few samples are needed before a stable result is found)
    """
    p = 0
    for _ in range(samples):
        disk = random_writes(disk_size, writes)
        matrix = consecutive_change_dict_to_matrix(construct_consecutive_changes_dict(disk))
        expected_chains = np.sum(matrix[:, 1]) / writes
        p += expected_chains * matrix[0, 2]

    return p / samples