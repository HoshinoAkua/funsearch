import numpy as np 
def priority(item: float, bins: np.ndarray) -> np.ndarray:
    # This heuristic prioritizes bins with the least remaining space after the item is placed,
    # but also takes into account the current occupancy of the bin.
    # The idea is to try to fill up the bins as evenly as possible,
    # but still prioritizing bins that will be closer to being full after adding the item.
    
    # First, we calculate the remaining space in each bin after adding the item.
    remaining_capacity = bins - item
    
    # Secondly, we calculate the inverse of the current space in each bin.
    # This will give a higher value for bins that are currently less filled.
    inverse_current_capacity = 1 / (bins + 1e-5)
    
    # Finally, we score each bin by multiplying the remaining space by the inverse of the current space.
    # This gives a higher score to bins that will have less remaining space after adding the item,
    # and also gives a higher score to bins that are currently less filled.
    score = remaining_capacity * inverse_current_capacity
    
    # We also add a small random noise to the score to slightly diversify the bin selection process
    # when there are multiple bins with similar scores.
    random_noise = np.random.normal(loc=0, scale=1e-5, size=bins.shape)
    score += random_noise

    return score