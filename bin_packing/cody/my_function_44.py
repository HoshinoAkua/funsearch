import numpy as np 
def priority(item: float, bins: np.ndarray) -> np.ndarray:
    # This heuristic is a modification of the prior heuristics. It prioritizes bins that would be filled closest to
    # their capacity without exceeding it, and maximizes the usage of the bins.
    # The score is computed as the product of two factors:
    # 1. The remaining space in the bin after adding the item (bins - item). The less remaining space, the higher the score.
    # 2. The inverse of the current occupancy of the bin (1 / (bins + 1e-5)). The less currently occupied, the higher the score.
    # We add an exponential function to give more weight to bins with less remaining space after adding the item.
    # This aims to minimize the number of bins used by filling each bin as much as possible before moving to the next.
    remaining_capacity = bins - item
    score = np.exp((bins - item) * (1 / (bins + 1e-5)))
    return score