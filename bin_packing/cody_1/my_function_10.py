import numpy as np 
def priority(item: float, bins: np.ndarray) -> np.ndarray:
    # This heuristic introduces a new concept of "Value to Size Ratio" (VSR). 
    # VSR is the ratio of the remaining space in the bin to the current size of the bin.
    # The idea is to prioritize bins with lower VSR, meaning the item fills a bigger proportion of the bin.
    # This should lead to a more efficient use of space, potentially reducing the total number of bins used.
    # Also, this heuristic provides less priority to bins which will be left with less than 10% of their capacity after adding the item.
    # This aims at avoiding the scenario where an item that could perfectly fit in a bin is not placed there because of previous decisions.

    # Calculate the leftover space in the bin after placing the item
    leftovers = bins - item

    # Calculate the value to size ratio. Add a small constant to avoid division by zero
    vsr = leftovers / (bins + 1e-10)

    # Assign high priority to bins with lower VSR
    vsr_score = 1 / (vsr + 1e-10)

    # Assign low priority to bins which will be left with less than 10% of their capacity after adding the item
    low_space_score = (leftovers < 0.1 * bins).astype(float)

    # Combine the scores. We give higher weight to vsr_score as we consider it more important
    # to fill the bins to a greater extent, while penalizing low_space_score to avoid leaving too many bins partially full
    return 0.9 * vsr_score - 0.1 * low_space_score