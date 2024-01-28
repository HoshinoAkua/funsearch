import numpy as np 
def priority(item: float, bins: np.ndarray) -> np.ndarray:
    #This heuristic prioritizes bins where the item will leave the least leftover space
    #but also takes into account the total capacity of the bin. The idea is to balance 
    #filling up bins efficiently while also not unduly prioritizing smaller bins, 
    #which might lead to a higher total number of bins used.

    # Calculate the leftover space after placing the item in each bin
    leftovers = bins - item

    # Look for bins where the item would leave a small amount of space remaining
    # We add a small constant to avoid division by zero
    efficiency_score = 1 / (leftovers + 1e-10)

    # Also take into account the total bin capacity. Larger bins get a higher score
    # Normalize the capacities so that they are in the same range as the efficiency scores
    capacity_score = bins / np.max(bins)

    # Combine the scores. We can adjust the weight of the efficiency_score and capacity_score
    # to make the heuristic more or less aggressive
    return 0.7 * efficiency_score + 0.3 * capacity_score