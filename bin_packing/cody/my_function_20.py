import numpy as np 
def priority(item: float, bins: np.ndarray) -> np.ndarray:
    # This heuristic operates on the principle of 'Best Fit'. It prioritizes the bin in which the item fits the most snugly, 
    # leaving the least remaining space. This is done by just subtracting the item size from the remaining capacity of each bin. 
    # A lower score indicates a better fit, as there is less wasted space. However, unlike the Best Fit heuristic, 
    # this heuristic applies a correction term, which is an inverse function of the current occupancy of the bin.
    # This encourages filling up bins to their maximum capacity while discouraging overuse of a single bin.
    # The small constant 1e-5 is added to avoid division by zero.
    return (bins - item) / (bins + 1e-5)