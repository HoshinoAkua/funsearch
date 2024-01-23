import numpy as np 
def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """Custom heuristic for online binpacking."""
    # Calculate the remaining space in each bin after placing the item.
    remaining_space = bins - item
    # Calculate the score for each bin.
    # Bins that are nearly full (remaining space close to 0) get high scores.
    # Bins that are nearly empty (remaining space close to bin size) also get high scores.
    # Other bins get scores between these two extremes.
    scores = np.abs(remaining_space - (bins / 2))
    return scores