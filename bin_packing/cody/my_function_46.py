import numpy as np 
def priority(item: float, bins: np.ndarray) -> np.ndarray:
    # This heuristic aims to maximize the utilization of each bin by prioritizing the bins which will have least remaining 
    # space after adding the item, and bins that are least used currently.
    # It achieves this by computing the score as the difference of two factors:
    # 1. The exponential of the remaining capacity in bin after adding the item (np.exp(bins - item)). The less remaining space,
    # the higher the score.
    # 2. The current bin usage rate, which is computed as the ratio of the total items already in the bin to the total number 
    # of bins (np.sum(bins)/len(bins)). The less currently used, the higher the score.
    # By doing this, the heuristic encourages filling of bins to their maximum capacity and ensures a balanced distribution of 
    # items across all bins, hence minimizing the total number of bins used.
    remaining_capacity = bins - item
    current_bin_usage_rate = np.sum(bins) / len(bins)
    score = np.exp(remaining_capacity) - current_bin_usage_rate
    return score