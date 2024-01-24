import numpy as np 
def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """Heuristic that prioritizes bins where the item fills a larger proportion of the remaining capacity."""
    return item / bins