import numpy as np 
def priority(item: float, bins: np.ndarray) -> np.ndarray:
    # The new heuristic gives high priority to bins where the item would fit perfectly 
    # and to bins where the item would fill the bin to nearly its full capacity (e.g., 90% capacity).
    # This should encourage more efficient use of bins, potentially reducing the total number of bins needed.

    # Calculate the leftover space and capacity ratio after placing the item in each bin
    leftovers = bins - item
    capacity_ratio = (bins - leftovers) / bins

    # Assign high priority to bins where the item would fit perfectly 
    perfect_fit_score = (leftovers == 0).astype(float)

    # Also assign high priority to bins where the item would fill the bin to 90% capacity or more
    high_fill_score = (capacity_ratio >= 0.9).astype(float)

    # Combine the scores. We give higher weight to high_fill_score as we consider it more important
    # to fill the bins to a greater extent
    return 0.5 * perfect_fit_score + 0.5 * high_fill_score