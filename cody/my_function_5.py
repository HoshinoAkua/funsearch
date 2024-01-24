import numpy as np 
def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """Heuristic that prioritizes bins based on the ratio of the item size to the bin capacity."""
    return item / bins