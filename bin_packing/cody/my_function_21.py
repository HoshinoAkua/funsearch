import numpy as np 
def priority(item: float, bins: np.ndarray) -> np.ndarray:
    # This heuristic combines the best of both the 'Best Fit' and 'Minimize Variance' approaches.
    # It computes a score for each bin based on how snugly the item would fit (the 'Best Fit' part), 
    # but it also takes into account the variance in bin sizes after adding the item (the 'Minimize Variance' part).
    # These two scores are then combined into a single score using a weighted average, with weights that could be adjusted.
    # For now, we will use equal weights (0.5 each) for simplicity.
    best_fit_score = (bins - item) / (bins + 1e-5)
    bin_sizes_after_adding_item = bins - item
    variance_score = 1 / (np.var(bin_sizes_after_adding_item) + 1e-5)
    return 0.5 * best_fit_score + 0.5 * variance_score