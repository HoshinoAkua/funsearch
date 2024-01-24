import numpy as np 
def priority(item: float, bins: np.ndarray) -> np.ndarray:
    # This heuristic aims to balance between bins that have the least remaining space after accommodating the item,
    # and bins that have the most remaining space currently.
    # First, we calculate the remaining space in each bin after adding the item.
    remaining_capacity = bins - item
    # Secondly, we calculate the ratio of the remaining space to the current space in each bin.
    capacity_ratio = remaining_capacity / (bins + 1e-5)
    # Finally, we score each bin by multiplying the remaining space by the capacity ratio.
    # This gives a higher score to bins that will have less remaining space after adding the item,
    # but also gives a higher score to bins that have more remaining space currently.
    # The idea is to try to fill up the bins as evenly as possible, while still prioritizing bins that can accommodate the item.
    score = remaining_capacity * capacity_ratio
    return score