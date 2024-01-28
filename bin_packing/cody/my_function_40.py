import numpy as np 
def priority(item: float, bins: np.ndarray) -> np.ndarray:
    # This heuristic is designed to prioritize bins that have least remaining space after the item has been added,
    # and also takes into account the variance of the remaining capacities among the bins.
    # It does so by giving higher scores to bins that will have less remaining space (after the item has been added),
    # and to bins that deviate less from the mean remaining capacity. 
    # The goal behind this heuristic is minimizing the total number of bins used while also keeping the variance of the bin usage low.
    remaining_capacity = bins - item
    mean_remaining_capacity = np.mean(remaining_capacity)
    # Variance of remaining capacities
    var_remaining_capacity = np.var(remaining_capacity)
    score = np.exp(-remaining_capacity) - np.abs(remaining_capacity - mean_remaining_capacity) / (var_remaining_capacity + 1e-5)
    return score