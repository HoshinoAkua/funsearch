import numpy as np 
def priority(item: float, bins: np.ndarray) -> np.ndarray:
    # This heuristic gives more priority to bins that have the least remaining space after adding the item.
    # It also takes into account bins that are less filled currently. This will discourage overuse of a single bin and also
    # helps in filling bins to their maximum capacity.
    # The heuristic score is a combination of two factors:
    # 1. The remaining space in the bin after adding the item (bins - item). The less remaining space, the higher the score.
    # 2. The inverse of the current occupancy of the bin (1 / (bins + item + 1e-5)). The less currently occupied, the higher the score.
    # Adding the item to the denominator in the inverse of current occupancy ensures that the bins that can just accommodate the item get higher priority.
    return (bins - item) * (1 / (bins + item + 1e-5))