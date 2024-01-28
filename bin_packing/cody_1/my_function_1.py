import numpy as np 
def priority(item: float, bins: np.ndarray) -> np.ndarray:
    # This heuristic gives priority to bins where the item will fit perfectly or 
    # where the item would fill the bin to a certain threshold (e.g., 80% capacity). 
    # The rationale is to prioritize filling bins to a useful extent over leaving 
    # little space remaining. This should encourage more balanced use of bins, 
    # potentially reducing the total number of bins needed.

    # Calculate the leftover space and capacity ratio after placing the item in each bin
    leftovers = bins - item
    capacity_ratio = (bins - leftovers) / bins

    # Assign high priority to bins where the item would fit perfectly 
    perfect_fit_score = (leftovers == 0).astype(float)

    # Also assign high priority to bins where the item would fill the bin to 80% capacity or more
    high_fill_score = (capacity_ratio >= 0.8).astype(float)

    # Combine the scores. We give higher weight to perfect_fit_score as we consider it more important
    return 0.6 * perfect_fit_score + 0.4 * high_fill_score