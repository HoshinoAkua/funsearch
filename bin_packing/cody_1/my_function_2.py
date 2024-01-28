import numpy as np 
def priority(item: float, bins: np.ndarray) -> np.ndarray:
    # In this heuristic, we prioritize bins based on the leftover space after placing the item in each bin.
    # However, instead of simply prioritizing the smallest leftover space, we instead favor bins where the leftover 
    # space is closest to the size of the smallest item not yet placed in a bin. This can help to avoid situations 
    # where a small item cannot be placed because all bins have leftover space that is slightly too small.

    # Calculate the leftover space after placing the item in each bin
    leftovers = bins - item

    # Calculate the difference between the leftover space and the size of the smallest item
    # For now, we assume that the smallest item has a size of 1.0. If information about the sizes of the other items 
    # was available, we could use that instead.
    diff = np.abs(leftovers - 1.0)

    # Convert the differences into scores. Bins where the leftover space is closest to the size of the smallest item 
    # get the highest score.
    scores = 1 / (diff + 1e-10)

    return scores