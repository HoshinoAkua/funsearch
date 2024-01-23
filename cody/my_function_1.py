import numpy as np 
def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """Heuristic that prioritizes bins based on the square root of the difference between the bin capacity and the item size."""
    return np.sqrt(bins - item)