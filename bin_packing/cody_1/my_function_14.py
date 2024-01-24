import numpy as np 
def priority(item: float, bins: np.ndarray) -> np.ndarray:
    # This heuristic prioritizes bins that are close to being filled after including the item.
    # It also prioritizes bins that will be over 50% filled after adding the item, since we want to avoid
    # having too many bins that are only partially filled.
    # The aim is to balance between filling up bins as much as possible and not having too many partially filled bins.

    # Calculate the remaining space after placing the item in each bin
    remaining_space = bins - item

    # Calculate the percentage of the bin that would be filled after adding the item
    fill_ratio = (bins - remaining_space) / bins

    # Assign a high priority to bins that will be almost filled after adding the item
    high_fill_score = (remaining_space < 0.1 * bins).astype(float)

    # Assign some priority to bins that will be over 50% filled after adding the item
    over_half_filled_score = (fill_ratio > 0.5).astype(float)

    # Combine the scores, weighing more towards high_fill_score
    return 0.7 * high_fill_score + 0.3 * over_half_filled_score