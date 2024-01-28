import numpy as np 
def priority(item: float, bins: np.ndarray) -> np.ndarray:
    # This heuristic prioritizes bins based on two factors: the remaining capacity after adding the item, and the total space used in the bin.
    # The idea is to prefer bins that will have least remaining space after adding the item (to fill the bins as much as possible), 
    # and also to prioritize bins that have been used less (to distribute items more evenly among the bins).
    # The score is computed as the product of the remaining capacity and the inverse of the total space used in the bin (plus a small constant to avoid division by zero).
    # The addition of the exponential function is to increase the difference between the scores, providing a more decisive decision when choosing bins.
    remaining_capacity = bins - item
    total_space_used = np.sum(bins) 
    score = np.exp(remaining_capacity) * (1/ (total_space_used + 1e-5))
    return score