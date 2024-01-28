import numpy as np 
def priority(item: float, bins: np.ndarray) -> np.ndarray:
    # This heuristic is designed to prioritize bins that will leave the smallest remaining capacity after the item has been added,
    # but also takes into consideration the variance among the remaining capacities of the bins. 
    # It does so by giving higher scores to bins that will leave smaller remaining capacity (after the item has been added),
    # and lower scores to bins that would result in a high variance of remaining capacities among the bins. 
    # The idea behind this heuristic is to strive for a balance between filling up the bins as much as possible (to minimize the total number of bins used),
    # and maintaining a small variance of remaining capacities among the bins (to avoid having a few bins that are almost full while many others are still empty).
    remaining_capacity = bins - item
    avg_remaining_capacity = np.mean(remaining_capacity)
    var_remaining_capacity = np.var(remaining_capacity)
    score = np.exp(-remaining_capacity) - np.square(remaining_capacity - avg_remaining_capacity) / (var_remaining_capacity + 1e-5)
    return score