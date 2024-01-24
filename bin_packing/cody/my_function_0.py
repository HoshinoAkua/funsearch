import numpy as np 
def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """Compute the priority of each bin for packing an item.
    
    Args:
    item: float, size of item to be added to the bin
    bins: numpy array, an array of capacities for each bin
    
    Returns:
    numpy array: array of priority scores for each bin
    """
    # Define a score function that rewards bins that will be more filled after packing the item.
    scores = bins - item
    # Compute a bonus for bins that will be exactly filled by the item.
    bonus = np.where(scores == 0, 1, 0)
    # Combine base scores and bonus and invert so that higher scores are better.
    return -(scores - bonus)