import numpy as np 
def priority(item: float, bins: np.ndarray) -> np.ndarray:
    # This heuristic gives priority to bins where the item would fit perfectly 
    # and to bins where the item would fill the bin to nearly its full capacity (e.g., 95% capacity).
    # This should encourage more efficient use of bins, potentially reducing the total number of bins needed.
    # In addition, it also gives a low priority to bins that would be less than 50% full after placing the item.
    # This aims to avoid leaving many bins partially full, which could lead to inefficient use of space.

    # Calculate the leftover space and capacity ratio after placing the item in each bin
    leftovers = bins - item
    capacity_ratio = (bins - leftovers) / bins

    # Assign high priority to bins where the item would fit perfectly 
    perfect_fit_score = (leftovers == 0).astype(float)

    # Also assign high priority to bins where the item would fill the bin to 95% capacity or more
    high_fill_score = (capacity_ratio >= 0.95).astype(float)
    
    # Assign low priority to bins where the item would fill the bin to less than 50% capacity
    low_fill_score = (capacity_ratio < 0.5).astype(float)

    # Combine the scores. We give higher weight to perfect_fit_score and high_fill_score as we consider it more important
    # to fill the bins to a greater extent, while penalizing low_fill_score to avoid leaving too many bins partially full
    return 0.4 * perfect_fit_score + 0.4 * high_fill_score - 0.2 * low_fill_score