import numpy as np 
def priority(item: float, bins: np.ndarray) -> np.ndarray:
    # This heuristic combines aspects of the previous two heuristics.
    # It prioritizes bins that will have the least remaining space after accommodating the item (similar to heuristic 2).
    # It also takes into account the variance of the remaining capacities after adding the item (similar to heuristic 1).
    # The aim is to balance the load across bins, while also filling up the bins as much as possible.
    # To combine these two aspects, a weighted harmonic mean is used, which gives a higher score if both aspects are high.
    remaining_space = bins - item 
    remaining_space_scores = remaining_space / (np.sum(remaining_space) + 1e-5)
    remaining_capacities_after_adding_item = bins - item
    variance_scores = 1 / (np.var(remaining_capacities_after_adding_item) + 1e-5)
    # Harmonic mean is used to combine the scores
    return 2 / ((1 / (remaining_space_scores + 1e-5)) + (1 / (variance_scores + 1e-5)))