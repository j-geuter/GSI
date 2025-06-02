import numpy as np

def group_stats(values, group_size=10):
    """
    Splits the input list 'values' into groups of size 'group_size' (the last group may be shorter),
    computes the mean for each group, then returns the overall mean and standard deviation of the group means.
    """
    group_means = []
    for i in range(0, len(values), group_size):
        group = values[i:i+group_size]
        group_means.append(np.mean(group))
    overall_mean = np.mean(group_means)
    overall_std = np.std(group_means)
    return overall_mean, overall_std
