import numpy as np 
def priority(item: float, bins: np.ndarray) -> np.ndarray:
    # This heuristic takes a slightly different approach. It still aims to 
    # minimize the variance of bin capacities after placing the item, but 
    # it also tries to minimize the maximum remaining capacity in any bin.
    # This encourages the algorithm to fill bins as fully as possible, 
    # without leading to a high variance in bin capacities.
    # The final score for each bin is a weighted combination of these two metrics,
    # with a preference for minimizing the maximum remaining capacity.
    bin_sizes_after_adding_item = bins - item
    variance = np.var(bin_sizes_after_adding_item)
    max_remaining_capacity = np.max(bin_sizes_after_adding_item)
    return 0.7 * (1 / (max_remaining_capacity + 1e-5)) + 0.3 * (1 / (variance + 1e-5))