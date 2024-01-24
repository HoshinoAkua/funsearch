import numpy as np 
def priority(item: float, bins: np.ndarray) -> np.ndarray:
    # This heuristic tries to balance the load among the bins, while trying to fill up the bins as much as possible.
    # It prioritizes bins that will have the least remaining space after accommodating the item.
    # It also takes into account the number of items already in the bin. The fewer items in the bin, the higher the priority.
    # This encourages the algorithm to distribute items between bins and not to overload any particular bin.
    # The final score is the harmonic mean of the two factors.
    remaining_space = bins - item 
    remaining_space_scores = remaining_space / (np.sum(remaining_space) + 1e-5) 
    num_items = len(bins) # assume len(bins) gives the number of items in the bin
    num_items_scores = 1 / (num_items + 1e-5) # the fewer items in the bin, the higher the score
    # Harmonic mean is used to combine the scores
    return 2 / ((1 / (remaining_space_scores + 1e-5)) + (1 / (num_items_scores + 1e-5)))