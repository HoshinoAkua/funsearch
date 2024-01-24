import numpy as np 
def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """Heuristic that prioritizes bins based on the absolute difference between the item size and the bin capacity."""
    return abs(bins - item)