import numpy as np 
def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """Heuristic that prioritizes bins with capacities closest to the average capacity."""
    avg_capacity = np.average(bins)
    return -np.abs(bins - avg_capacity)