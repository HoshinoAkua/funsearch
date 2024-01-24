import numpy as np 
def priority(item: float, bins: np.ndarray) -> np.ndarray:
    # This heuristic is designed to balance between filling up the bins as much as possible 
    # and minimizing the deviation from the average remaining capacity among the bins.
    # It adds a new factor to the score: the ratio of the remaining capacity of a bin (after accommodating the item) 
    # and the average remaining capacity among the bins. 
    # Bins that have a lower ratio (i.e., their remaining capacity is below average) get a higher score, 
    # which encourages the algorithm to use bins that are less full.
    remaining_capacity = bins - item
    avg_remaining_capacity = np.mean(remaining_capacity)
    score = np.exp(-remaining_capacity) * ((remaining_capacity / (avg_remaining_capacity + 1e-5)) ** 2)
    return score