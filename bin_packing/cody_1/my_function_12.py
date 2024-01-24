import numpy as np 
def priority(item: float, bins: np.ndarray) -> np.ndarray:
    # This heuristic tries to maximize the utilization of the bins by prioritizing 
    # bins where the item would increase the fill level to above 70%, but not overfill it. 
    # Moreover, it gives higher priority to bins where the item would leave less leftover space.

    # Calculate the leftover space and capacity ratio after placing the item in each bin
    leftovers = bins - item
    capacity_ratio = (bins - leftovers) / bins

    # Assign high priority to bins where the item would increase the fill level to above 70%, 
    # but not overfill it.
    utilization_score = ((capacity_ratio >= 0.7) & (capacity_ratio <= 1)).astype(float)

    # Look for bins where the item would leave a small amount of space remaining
    # We add a small constant to avoid division by zero
    efficiency_score = 1 / (leftovers + 1e-10)

    # Combine the scores. We give higher weight to utilization_score as we consider it more important
    return 0.7 * utilization_score + 0.3 * efficiency_score