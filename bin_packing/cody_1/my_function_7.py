import numpy as np 
def priority(item: float, bins: np.ndarray) -> np.ndarray:
    # This heuristic prioritizes bins where the item will fill the bin to a higher capacity 
    # and where the item will leave a small amount of space remaining after being placed. 
    # The idea is to balance filling up bins effectively while also minimizing the leftover space.

    # Calculate the leftover space after placing the item in each bin
    leftovers = bins - item

    # Look for bins where the item would leave a small amount of space remaining
    # We add a small constant to avoid division by zero
    efficiency_score = 1 / (leftovers + 1e-10)

    # Also assign high priority to bins where the item would fill the bin to 70% capacity or more.
    # The capacity ratio is calculated as the ratio of the space that will be occupied by the item to the total capacity of the bin.
    capacity_ratio = item / bins
    high_fill_score = (capacity_ratio >= 0.7).astype(float)

    # Combine the scores. We give higher weight to efficiency_score as we consider it more important
    return 0.6 * efficiency_score + 0.4 * high_fill_score