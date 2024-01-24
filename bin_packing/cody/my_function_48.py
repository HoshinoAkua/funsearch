import numpy as np 
def priority(item: float, bins: np.ndarray) -> np.ndarray:
    # This heuristic is designed to prioritize bins that will have less remaining capacity after the item has been added,
    # while minimizing the standard deviation of remaining capacities among the bins.
    # It includes a penalty term for bins that would be overfilled by the item.
    # The idea behind this heuristic is to balance the need to fill up the bins as much as possible (to minimize the total number of bins used)
    # and the need to distribute the items evenly among the bins (to avoid the scenario where some bins are almost full while others are almost empty).
    remaining_capacity = bins - item
    avg_remaining_capacity = np.mean(remaining_capacity)
    std_remaining_capacity = np.std(remaining_capacity)
    overfill_penalty = np.where(remaining_capacity < 0, np.inf, 0)
    score = np.exp(-remaining_capacity) - (np.abs(remaining_capacity - avg_remaining_capacity) / (std_remaining_capacity + 1e-5)) + overfill_penalty
    return score