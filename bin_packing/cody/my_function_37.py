import numpy as np 
def priority(item: float, bins: np.ndarray) -> np.ndarray:
    # This heuristic aims to prioritize bins considering two key factors: the remaining capacity of the bin after adding the item and the overall usage of bins.
    # The idea is to encourage filling up the bins as much as possible while also promoting a uniform distribution of items across all bins.
    # The score is computed by first considering the remaining space in each bin after the item has been added. This is done to prefer bins that will be almost full after accommodating the item.
    # The second part of the score computation takes into consideration the total space used in all bins, aiming to distribute items more evenly among the bins.
    # A logarithmic function is used to dampen the large remaining capacity values, giving a chance to bins with smaller remaining capacities.
    # Additionally, a normalization factor is added to prevent division by zero.
    remaining_capacity = bins - item
    total_space_used = np.sum(bins) 
    score = np.log(remaining_capacity + 1) * (1/ (total_space_used + 1e-5))
    return score