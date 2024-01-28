import numpy as np 
def priority(item: float, bins: np.ndarray) -> np.ndarray:
    # This heuristic gives preference to bins that will be least filled after accommodating the item.
    # The heuristic score is the inverse of the bin size after adding the item.
    # This encourages distribution of items across bins while trying to fill each bin as much as possible.
    return 1 / (bins - item + 1e-5)