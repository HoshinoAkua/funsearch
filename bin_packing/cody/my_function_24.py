import numpy as np 
def priority(item: float, bins: np.ndarray) -> np.ndarray:
    # This heuristic is designed to prioritize bins that have less remaining capacity after the item has been added,
    # but also take into account the standard deviation of the remaining capacities among the bins.
    # It does so by giving higher scores to bins that have less remaining capacity (after the item has been added),
    # and lower scores to bins that would significantly deviate from the average remaining capacity if they were to accommodate the item.
    # The idea behind this heuristic is to achieve a balance between filling up the bins as much as possible (to minimize the total number of bins used),
    # and distributing the items evenly among the bins (to avoid having a few bins that are almost full while many others are still empty).
    remaining_capacity = bins - item
    avg_remaining_capacity = np.mean(remaining_capacity)
    std_remaining_capacity = np.std(remaining_capacity)
    score = np.exp(-remaining_capacity) - np.abs(remaining_capacity - avg_remaining_capacity) / (std_remaining_capacity + 1e-5)
    return score