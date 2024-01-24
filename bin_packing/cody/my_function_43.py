import numpy as np 
def priority(item: float, bins: np.ndarray) -> np.ndarray:
    # This heuristic is a combination of example heuristic 1 and 2.
    # It prioritizes bins that will leave a smaller remaining capacity after the item has been added,
    # and also aims to distribute items evenly among the bins.
    # The idea is to strike a balance between minimizing the number of bins used and promoting a uniform distribution of items among bins.
    # In the first part of the score computation, it gives a higher priority to the bins that will be almost full after accommodating the item.
    # In the second part, it rewards bins that will contribute to a smaller variance in remaining capacities among the bins.
    # A logarithmic function and an exponential function are used to dampen large remaining capacity values and to amplify the effect of smaller remaining capacities.
    remaining_capacity = bins - item
    total_space_used = np.sum(bins)
    avg_remaining_capacity = np.mean(remaining_capacity)
    var_remaining_capacity = np.var(remaining_capacity)
    
    score = np.log(remaining_capacity + 1) * (1 / (total_space_used + 1e-5)) + np.exp(-remaining_capacity) - np.square(remaining_capacity - avg_remaining_capacity) / (var_remaining_capacity + 1e-5)
    return score