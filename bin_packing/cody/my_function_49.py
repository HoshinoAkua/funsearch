import numpy as np 
def priority(item: float, bins: np.ndarray) -> np.ndarray:
    # This heuristic is a hybrid of example heuristic 1 and 2 with an additional component.
    # It prioritizes bins that will have the smallest remaining capacity after the item has been added.
    # It also includes a factor to reward bins that will contribute to a smaller variation in remaining capacities among the bins.
    # Lastly, it includes a factor that rewards bins that are less filled, promoting a more uniform distribution of items among bins.
    # These three components are combined using a weighted sum, where the weights reflect the relative importance of each component.
    remaining_capacity = bins - item
    total_space_used = np.sum(bins)
    avg_remaining_capacity = np.mean(remaining_capacity)
    var_remaining_capacity = np.var(remaining_capacity)
    
    # Component 1: Smallest remaining capacity
    comp1 = np.exp(-remaining_capacity)
    # Component 2: Smaller variation in remaining capacities
    comp2 = -np.square(remaining_capacity - avg_remaining_capacity) / (var_remaining_capacity + 1e-5)
    # Component 3: More uniform distribution of items
    comp3 = 1 / (total_space_used + 1e-5)
    
    # Combine components using a weighted sum
    weights = np.array([0.5, 0.3, 0.2])
    score = weights[0] * comp1 + weights[1] * comp2 + weights[2] * comp3
    return score