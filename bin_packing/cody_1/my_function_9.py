import numpy as np 
def priority(item: float, bins: np.ndarray) -> np.ndarray:
    # This heuristic introduces a new concept of "optimal usage factor" which is calculated as 
    # the ratio of the item size and the bin's current capacity.
    # The idea is to give highest priority to the bins where the item fits just perfectly or where 
    # the bin would be almost full after placing the item (i.e., optimal usage).
    # This should encourage the items to fill the bins as much as possible, 
    # potentially reducing the total number of bins needed.

    # Calculate the optimal usage factor for each bin
    optimal_usage_factor = item / bins

    # Normalize the optimal usage factor to get a value between 0 and 1 for each bin
    optimal_usage_factor_normalized = (optimal_usage_factor - optimal_usage_factor.min()) / (optimal_usage_factor.max() - optimal_usage_factor.min())

    # Calculate the priority score
    # Bins with the highest optimal usage factor (i.e., where the item fits best or makes the bin almost full) will have the highest priority
    priority = optimal_usage_factor_normalized

    return priority