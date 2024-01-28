import numpy as np 
def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Compute the priority of each bin for packing an item. The priority is determined by the following factors:
    - The remaining capacity of the bin after the item is added. Bins where the item would leave little unused space have higher priority.
    - The size of the item. Larger items have more priority.
    - Whether adding the item would result in the bin being exactly filled. Bins where the item would fit exactly have the highest priority.
    - The relative capacity of the bin compared to other bins.
    
    Args:
    item: float, size of item to be added to the bin
    bins: numpy array, an array of capacities for each bin
    
    Returns:
    numpy array: array of priority scores for each bin
    """
    # Base score function that rewards bins that will have less remaining space after packing the item.
    scores = bins - item
    
    # Compute a bonus for bins where the item would fit exactly. The bonus is greater if the item is larger.
    exact_fit_bonus = np.where(scores == 0, 2 + item / max(bins), 0)
    
    # Additional priority for larger items.
    large_item_bonus = np.where(item >= np.mean(bins), 1, 0)
    
    # Additional priority for bins with more relative capacity.
    relative_capacity_bonus = bins / max(bins)
    
    # Combine base scores and bonuses.
    return -(scores - exact_fit_bonus - large_item_bonus - relative_capacity_bonus)