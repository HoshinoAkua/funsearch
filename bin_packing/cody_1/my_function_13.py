import numpy as np 
def priority(item: float, bins: np.ndarray) -> np.ndarray:
    # This new heuristic gives high priority to bins where the item would fit perfectly 
    # and to bins where the item would fill the bin to nearly its full capacity (e.g., 85% capacity).
    # This should encourage more efficient use of bins, potentially reducing the total number of bins needed.
    # It also gives a lower priority to bins that would be less than 40% full after placing the item,
    # aiming to prevent having many bins that are just partially full, which could lead to inefficient use of space.

    # Calculate the leftover space and capacity ratio after placing the item in each bin
    leftovers = bins - item
    capacity_ratio = (bins - leftovers) / bins

    # Assign high priority to bins where the item would fit perfectly 
    perfect_fit_score = (leftovers == 0).astype(float)

    # Also assign high priority to bins where the item would fill the bin to 85% capacity or more
    high_fill_score = (capacity_ratio >= 0.85).astype(float)
    
    # Assign low priority to bins where the item would fill the bin to less than 40% capacity
    low_fill_score = (capacity_ratio < 0.4).astype(float)

    # Combine the scores. We give higher weight to perfect_fit_score and high_fill_score as we consider it more important
    # to fill the bins to a greater extent, while penalizing low_fill_score to avoid leaving too many bins partially full
    return 0.5 * perfect_fit_score + 0.4 * high_fill_score - 0.1 * low_fill_score