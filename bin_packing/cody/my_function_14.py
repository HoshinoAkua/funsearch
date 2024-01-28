import numpy as np 
def priority(item: float, bins: np.ndarray) -> np.ndarray:
    # This heuristic gives preference to bins that will be closest to the average bin size after accommodating the item.
    # The heuristic score is an exponential function of the absolute difference between the bin size after adding the item
    # and the average bin size. This encourages equal distribution of items across bins.
    avg_bin_size = np.mean(bins)
    return np.exp(-np.abs((bins - item) - avg_bin_size))