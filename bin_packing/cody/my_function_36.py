import numpy as np 
def priority(item: float, bins: np.ndarray) -> np.ndarray:
    # This heuristic combines the ideas from the first and second heuristics, but adds a twist to improve packing efficiency.
    # It calculates the remaining capacity after the item has been added.
    # It calculates the average remaining capacity after the item has been added to all the bins.
    # The score of a bin is a combination of three factors:
    # 1. The inverse of the remaining capacity in the bin after adding the item. The less remaining space, the higher the score.
    # 2. The inverse square of the difference between the remaining capacity in the bin and the average remaining capacity. The smaller the difference, the higher the score.
    # 3. The inverse of the standard deviation of remaining capacities among the bins. The smaller the standard deviation, the higher the score.
    # The intuition behind this heuristic is to try to pack items into bins such that the remaining capacities of the bins are as uniform as possible,
    # which is likely to result in a more efficient packing scheme.
    remaining_capacity = bins - item
    avg_remaining_capacity = np.mean(remaining_capacity)
    std_remaining_capacity = np.std(remaining_capacity)
    score = (1 / (remaining_capacity + 1e-5)) * (1 / ((np.abs(remaining_capacity - avg_remaining_capacity)) + 1e-5) ** 2) * (1 / (std_remaining_capacity + 1e-5))
    return score