import numpy as np 
def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """Heuristic that prioritizes bins based on the remaining capacity after adding the item."""
    return bins - item