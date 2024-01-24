import numpy as np 
def priority(item: float, bins: np.ndarray) -> np.ndarray:
    # This heuristic builds on the second example. It prioritizes bins that will be filled closest to their capacity without exceeding it.
    # The score is computed as the product of two factors:
    # 1. The remaining space in the bin after adding the item (bins - item). The less remaining space, the higher the score.
    # 2. The inverse square of the current occupancy of the bin (1 / (bins + 1e-5) ** 2). The less currently occupied, the higher the score.
    # This encourages the algorithm to fill up bins to their maximum capacity while discouraging overuse of a single bin.
    return (bins - item) * (1 / (bins + 1e-5) ** 2)