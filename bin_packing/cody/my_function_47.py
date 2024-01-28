import numpy as np 
def priority(item: float, bins: np.ndarray) -> np.ndarray:
    # This heuristic seeks to balance the desire for high bin utilisation with the need to avoid leaving bins almost empty.
    # The heuristic splits the problem into two parts:
    # 1. If adding the item would fill the bin to at least half its capacity, the bin is scored based on how full it would be.
    # 2. If adding the item would fill the bin to less than half its capacity, the bin is scored based on how empty it would be.
    # This approach encourages the heuristic to fill up bins that are already somewhat filled, while also seeking out bins that 
    # can be filled by the current item to a significant degree.
    remaining_capacity = bins - item
    mask = remaining_capacity <= 0.5 * bins
    score = np.empty_like(bins)
    score[mask] = 1 / (remaining_capacity[mask] + 1e-5)
    score[~mask] = bins[~mask] / (remaining_capacity[~mask] + 1e-5)
    return score