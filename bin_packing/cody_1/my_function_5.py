import numpy as np 
def priority(item: float, bins: np.ndarray) -> np.ndarray:
    # This heuristic prioritizes bins where the item would fit perfectly, 
    # and bins where the item would leave the least amount of empty space.

    # Calculate the leftover space after placing the item in each bin
    leftovers = bins - item

    # Assign high priority to bins where the item would fit perfectly 
    perfect_fit_score = (leftovers == 0).astype(float)

    # Also assign high priority to bins where the item would leave the least amount of empty space
    # By minimizing the leftover space, we aim to achieve a more efficient use of bin spaces
    min_leftover_score = (1 - leftovers / bins)

    # Combine the scores. We give equal weight to both factors
    return 0.5 * perfect_fit_score + 0.5 * min_leftover_score