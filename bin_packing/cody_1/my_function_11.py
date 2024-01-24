import numpy as np 
def priority(item: float, bins: np.ndarray) -> np.ndarray:
    # The new heuristic introduces a concept of "Efficient Utilization Score" (EUS). 
    # EUS is a score that measures how efficiently the space of the bin is used if the current item is placed in it.
    # The score is calculated as the ratio of the size of the item to the current size of the bin, squared. 
    # The square is used to give more priority to bins where the item fills a larger proportion of the bin.
    # Additionally, this heuristic penalizes bins which will be left with less than 20% of their capacity after adding the item.
    # This discourages the use of bins which will be nearly full but not quite, leading to more efficient use of space overall.

    # Calculate the leftover space in the bin after placing the item
    leftovers = bins - item

    # Calculate the Efficient Utilization Score
    eus = (item / (bins + 1e-10))**2

    # Assign low priority to bins which will be left with less than 20% of their capacity after adding the item
    low_space_penalty = (leftovers < 0.2 * bins).astype(float)

    # Combine the scores. We give higher weight to eus as we consider it more important
    # to efficiently utilize the space in the bins, while penalizing low_space_penalty to avoid leaving too many bins nearly full
    return eus - 0.1 * low_space_penalty