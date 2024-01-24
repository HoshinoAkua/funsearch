import numpy as np 
def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """Heuristic that prioritizes bins with least remaining space after adding the item."""
    remaining_space = bins - item
    return -remaining_space