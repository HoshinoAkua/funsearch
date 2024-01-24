import numpy as np 
def priority(item: float, bins: np.ndarray) -> np.ndarray:
    # The heuristic prefers bins with smaller remaining space after accommodating the item
    # This encourages filling up bins to their maximum capacity before using a new one
    return 1 / (bins - item + 1e-5) # Add a small constant to avoid division by zero