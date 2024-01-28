import numpy as np 
def priority(item: float, bins: np.ndarray) -> np.ndarray:
    # This heuristic will prioritize bins which will have least remaining space after adding the item,
    # and also bins which are least filled currently. 
    # The first part ensures that bins are filled as much as possible, reducing the total number of bins.
    # The second part ensures that the items are distributed evenly among the bins, preventing a scenario
    # where some bins are completely filled while others are empty.
    # The score is computed as the sum of two factors:
    # 1. The inverse of the remaining space in the bin after adding the item (1 / (bins - item + 1e-5)). The less remaining space, the higher the score.
    # 2. The inverse of the current occupancy of the bin (1 / (bins + 1e-5)). The less currently occupied, the higher the score.
    # The +1e-5 is to prevent division by zero errors.
    
    remaining_capacity_after_item = bins - item
    score = (1 / (remaining_capacity_after_item + 1e-5)) + (1 / (bins + 1e-5))
    return score