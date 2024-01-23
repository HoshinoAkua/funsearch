import numpy as np 
def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """Heuristic that prioritizes bins with the least remaining capacity and closest to the item size."""
    bins_capacity_diff = bins - item
    avg_diff = np.average(bins_capacity_diff)
    return -np.abs(bins_capacity_diff - avg_diff)