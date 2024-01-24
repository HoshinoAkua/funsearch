import numpy as np 
def priority(item: float, bins: np.ndarray) -> np.ndarray:
    # This heuristic gives preference to bins that will have the least remaining space after accommodating the item.
    # This encourages filling up bins to their maximum capacity before moving on to the next.
    return 1 / (bins - item + 1e-9)