import numpy as np 
def priority(item: float, bins: np.ndarray) -> np.ndarray:
    # This heuristic prefers bins that are nearly full after accommodating the item,
    # and also prefers bins that are almost empty. This is to encourage balance between
    # fully utilizing bins and distributing items evenly across bins.
    # The heuristic score is the maximum of two inversely proportional functions:
    # 1 / (bins - item + 1e-5) and 1 / (bins + 1e-5). The first term is high when the bin
    # is nearly full after the item is added, and the second term is high when the bin is almost empty.
    return np.maximum(1 / (bins - item + 1e-5), 1 / (bins + 1e-5)) # Add a small constant to avoid division by zero