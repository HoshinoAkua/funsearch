import numpy as np 
def priority(item: float, bins: np.ndarray) -> np.ndarray:
  def s(bin, item):
    diff = bin - item
    if diff <= 0:
      return 0
    elif diff <= 1:
      return 1
    else:
      return 1 / diff

  return np.array([s(b, item) for b in bins])