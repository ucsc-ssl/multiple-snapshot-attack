import math
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

def get_clean_chains(disk, target_changes, runs=10):
    """
    Constructs a consecutive change dictionary based on the disk provided.
    It aims to sample chains such that there are target_changes number of changes.
    It does this runs times and takes the disk closest to the targeted number of changes.

    disk: real disk to sample from
    target_changes: number of changes to make
    runs: number of times to sample
    """
    disk_cumulative_prob = np.cumsum(disk[:, 2])
    disk_cumulative_prob[-1] = 1.0

    dicts = []
    min_idx = 0
    min_changes = np.inf
    for i in range(runs):
        changes = 0
        consecutive_change_dict = {}

        while changes < target_changes:
            rand = np.random.rand()
            idx = np.argmax(disk_cumulative_prob >= rand)
            chain = disk[idx, 0]

            if chain in consecutive_change_dict:
                consecutive_change_dict[chain] += 1
            else:
                consecutive_change_dict[chain] = 1

            changes += chain

        if changes < min_changes:
            min_idx = i
            min_changes = changes

        dicts.append(consecutive_change_dict)
    return dicts[min_idx]

def add_artifice_ones(consecutive_change_dict, artifice_changes):
    """
    Adds artifice_changes number of 1s to consecutive_change_dict.

    consecutive_change_dict: a consecutive change dict
    artifice_changes: the number of occurences of singletons to add
    """
    if 1 in consecutive_change_dict:
        consecutive_change_dict[1] += artifice_changes
    else:
        consecutive_change_dict[1] = artifice_changes

    return consecutive_change_dict

def gen_data(in_data, clean_samples, clean_changes, artifice_samples, public_changes, artifice_singletons):
    """
    From sample disks, in_data, it generates clean_samples number of chain dictionaries records targetting clean_changes number of changes.
    Then it adds artifice_samples number of of chain dictionaries targeting public_changes number of clean chains, with an additional artifice_singletons number of singletons.
    It reports the truth for each sample with a 0 indicating a clean disk and a 1 indicating a disk with Artifice.

    in_data: an array of disk data to sample from
    clean_samples: number of clean disks in the dataset
    clean_changes: target changes for clean samples
    artifice_samples: number of simulated disks with an artifice volume
    public_changes: number of public changes to target for artifice disks
    artifice_singletons: number of artifice singletons for artifice disks
    """
    data = []
    truth = []

    i = 0
    while i < clean_samples:
        disk = in_data[i % len(in_data)]
        data.append(get_clean_chains(disk, clean_changes))
        truth.append(0)
        i += 1

    i = 0
    while i < artifice_samples:
        disk = in_data[i % len(in_data)]
        dict = get_clean_chains(disk, public_changes[i % len(public_changes)])
        dict = add_artifice_ones(dict, artifice_singletons[i % len(artifice_singletons)])
        data.append(dict)
        truth.append(1)
        i += 1

    return data, np.array(truth)

def construct_features(data):
    """
    Construct features from a dataset, data. The feature is simply the probability of a singleton on a disk.

    data: the dataset
    """
    features = np.zeros(len(data))

    for i in range(len(data)):
        total = np.sum(list(data[i].values()))
        num_singletons = 0 if not 1 in data[i].keys() else data[i][1]
        features[i] = num_singletons / total
    return features.reshape(-1, 1)

def train_lr(data, truth):
    """
    Returns a trained logistic regression classifier. It uses data as the features and truth as the truth labels.

    data: feature to train on
    truth: truth vector (1 is Artifice, 0 is no Artifice)
    """
    log_r = LogisticRegression()
    log_r.fit(data, truth)

    return log_r

def metrics(ind, pred, true):
    """
    Takes an index, idx, a list of predictions, pred, and a list of truth values, true, and produces classifier metrics.

    ind: index of the predictions (used to identify the size of Artifice instance)
    pred: predictions
    true: truth vector
    """
    tn, fp, fn, tp = confusion_matrix(true, pred).ravel()
    acc = (tp + tn) / (tn + fp + fn + tp)
    fpr = fp / (tn + fp)
    ppv = tp / (tp + fp) if tp + fp != 0 else 0.0
    fnr = fn / (fn + tp)
    recall = tp / (tp + fn)

    return [ind, acc, ppv, recall, fpr , fnr]

def get_ci(data, means, z=1.96):
    """
    Computes the confidence interval for all results.

    data: the results from all runs
    means: the means for each metric for each size artifice instance
    z: confidence interval z value
    """
    n = math.sqrt(len(data))

    ci = np.zeros(means.shape)
    for r in range(ci.shape[0]):
        for c in range(ci.shape[1]):
            sample = np.array([d[r,c] for d in data])
            ci[r,c] = np.std(sample) / n

    return (ci * z)