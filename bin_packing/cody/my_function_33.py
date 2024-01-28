import numpy as np 
def priority(item: float, bins: np.ndarray) -> np.ndarray:
    # This heuristic is a combination of the first and second heuristics with a slight modification.
    # It prioritizes bins that will be filled closest to their capacity without exceeding it.
    # It also maintains a balance between filling up the bins as much as possible and distributing the items evenly among the bins.
    # The score is computed as the product of three factors:
    # 1. The remaining space in the bin after adding the item (bins - item). The less remaining space, the higher the score.
    # 2. The inverse square of the current occupancy of the bin (1 / (bins + 1e-5) ** 2). The less currently occupied, the higher the score.
    # 3. The absolute deviation from the average remaining capacity (abs(bins - item - avg_remaining_capacity)) The smaller the deviation, the higher the score.
    remaining_capacity = bins - item
    avg_remaining_capacity = np.mean(remaining_capacity)
    score = (bins - item) * (1 / (bins + 1e-5) ** 2) * np.exp(-np.abs(remaining_capacity - avg_remaining_capacity))
    return score