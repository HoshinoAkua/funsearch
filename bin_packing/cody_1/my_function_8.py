import numpy as np 
def priority(item: float, bins: np.ndarray) -> np.ndarray:
    # The new heuristic introduces the concept of a "slack factor", 
    # which is the difference between the bin's current capacity and the item size. 
    # Bins with the lowest slack factor (i.e., bins where the item fits most snugly) are given the highest priority.

    # Calculate the slack factor for each bin
    slack_factor = bins - item

    # Normalize the slack factor to get a value between 0 and 1 for each bin
    slack_factor_normalized = (slack_factor - slack_factor.min()) / (slack_factor.max() - slack_factor.min())

    # Calculate the inverse slack factor (i.e., the priority)
    # Bins with the lowest slack factor (i.e., where the item fits most snugly) will have the highest priority
    priority = 1 - slack_factor_normalized

    return priority