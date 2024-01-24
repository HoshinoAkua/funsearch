import numpy as np 
def priority(item: float, bins: np.ndarray) -> np.ndarray:
    # This heuristic gives priority to bins that will have the least remaining space after accommodating the item.
    # But it also considers the current occupancy of the bin. This encourages filling up bins to their maximum capacity 
    # while also maintaining a balance in the distribution of items across bins.
    # The score is computed as the product of two terms: 
    # 1. The remaining space in the bin after adding the item (bins - item). Lower the remaining space, higher the score. 
    # 2. The current occupancy of the bin (bins / np.sum(bins)). Higher the current occupancy, higher the score.
    return (bins - item) * (bins / np.sum(bins))