import numpy as np 
def priority(item: float, bins: np.ndarray) -> np.ndarray:
    # This heuristic gives preference to bins that will be closest to the full capacity after accommodating the item. 
    # The heuristic score is a sigmoid function of the absolute difference between the bin size after adding the item 
    # and the full capacity. This encourages filling up the bins as much as possible.
    full_capacity = np.max(bins)
    return 1 / (1 + np.exp(-(bins - item - full_capacity)))