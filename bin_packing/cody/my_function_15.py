import numpy as np 
def priority(item: float, bins: np.ndarray) -> np.ndarray:
    # This heuristic aims to minimize the variance of bin capacities after accommodating the item.
    # It computes the variance of bin sizes after adding the item, then returns the inverse variance.
    # Bins that would result in lower variance (more evenly distributed items) are given higher priority.
    # A small constant is added to the variance to avoid division by zero.
    bin_sizes_after_adding_item = bins - item
    variance = np.var(bin_sizes_after_adding_item)
    return 1 / (variance + 1e-5)