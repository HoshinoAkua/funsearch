import numpy as np 
def priority(item: float, bins: np.ndarray) -> np.ndarray:
  def s(bin, item):
    diff = bin - item
    if diff <= 0:
      return 0
    elif diff <= 1:
      return 0.9
    elif diff <= 2:
      return 0.8
    elif diff <= 3:
      return 0.7
    elif diff <= 4:
      return 0.6
    else:
      return 1 / diff

  return np.array([s(b, item) for b in bins])