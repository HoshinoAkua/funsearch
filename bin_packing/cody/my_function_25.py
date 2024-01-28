import numpy as np 
def priority(item: float, bins: np.ndarray) -> np.ndarray:
    # This heuristic is a modified version of 'example heuristic 2'. It combines the best of the 'Best Fit' and 'Minimize Variance' approaches.
    # However, instead of using a weighted average to combine the scores, this heuristic uses a multiplicative approach to make sure both scores are taken into account.
    # The 'variance_score' is re-calculated here, to consider the variance in remaining capacities of the bins after adding the item.
    # This attempts to balance the load across bins more effectively, by discouraging the algorithm from filling up any single bin too quickly.
    best_fit_score = (bins - item) / (bins + 1e-5)
    remaining_capacities_after_adding_item = bins - item
    variance_score = 1 / (np.var(remaining_capacities_after_adding_item) + 1e-5)
    return best_fit_score * variance_score