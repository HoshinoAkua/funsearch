import numpy as np 
def priority(item: float, bins: np.ndarray) -> np.ndarray:
    # This heuristic is designed to prioritize bins on the basis of how much they are filled after adding the item
    # and the deviation of the remaining space in the bin from the average remaining space in all bins.
    # By doing this, the algorithm tries to keep the distribution of space in all bins as even as possible 
    # while also trying to fill the bins as much as possible.
    remaining_capacity = bins - item
    average_remaining_capacity = np.mean(remaining_capacity)
    # The score is comprised of two factors:
    # 1. The negative of the remaining capacity in the bin after adding the item. The less the remaining capacity, the higher the score.
    # 2. The absolute deviation of the remaining capacity in the bin from the average remaining capacity in all bins. 
    #    The less the deviation, the higher the score.
    score = -remaining_capacity + np.abs(average_remaining_capacity - remaining_capacity)
    return score