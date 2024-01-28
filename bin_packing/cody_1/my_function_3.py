import numpy as np 
def priority(item: float, bins: np.ndarray) -> np.ndarray:
    # This heuristic tries to balance the remaining space in each bin after placing the item
    # and the amount of space that the item will occupy in the bin.

    # Calculate the leftover space after placing the item in each bin
    leftovers = bins - item

    # Calculate how much of the bin's capacity the item will fill
    fill_ratio = item / bins

    # Calculate the score for each bin. The score is based on the remaining space in the bin
    # after placing the item (we prefer bins with less leftover space) and the amount of space 
    # that the item will occupy in the bin (we prefer bins where the item will occupy a larger 
    # proportion of the bin's capacity). We use a weighted sum to combine these two factors into 
    # a single score. The weights can be adjusted to prioritize one factor over the other.
    return 0.5 * (1 / (leftovers + 1e-10)) + 0.5 * fill_ratio