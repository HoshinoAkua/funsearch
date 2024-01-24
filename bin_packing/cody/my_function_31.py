import numpy as np 
def priority(item: float, bins: np.ndarray) -> np.ndarray:
    # This heuristic is designed to prioritize bins that would have least remaining space after the item has been added,
    # and also takes into account the bins that are less used.
    # It does so by giving higher scores to bins that will have less remaining space (after the item has been added),
    # and higher scores to bins that were less used.
    # The idea behind this heuristic is to fill up the bins as much as possible (to minimize the total number of bins used),
    # and to distribute the items amongst less used bins (to avoid overusing some bins while others are still mostly empty).
    remaining_capacity = bins - item
    bin_usage = np.sum(bins) / len(bins)
    score = np.exp(-remaining_capacity) - np.abs(bin_usage - remaining_capacity)
    return score