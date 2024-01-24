import numpy as np 
def priority(item: float, bins: np.ndarray) -> np.ndarray:
    # This heuristic gives priority to bins that will have the least remaining space after accommodating the item,
    # but inversely proportionate to its current occupancy. This encourages filling up bins to their maximum capacity 
    # while discouraging overuse of a single bin.
    # The score is computed as the product of two terms: 
    # 1. The remaining space in the bin after adding the item (bins - item). Lower the remaining space, higher the score. 
    # 2. Inverse of the current occupancy of the bin (np.sum(bins) / bins). Lower the current occupancy, higher the score.
    # The small constant 1e-5 is added to avoid division by zero.
    return (bins - item) * (np.sum(bins) / (bins + 1e-5))