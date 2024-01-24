import numpy as np 
def priority(item: float, bins: np.ndarray) -> np.ndarray:
  """This heuristic is a variant of the previous heuristics.
  
  It prioritizes bins that will have the least remaining space after the new item is added while also considering the overall distribution of the remaining spaces in the bins. 
  
  It uses the remaining capacity in each bin after the item would be added, the average remaining capacity and the variance of the remaining capacities. 
  
  In addition to these, it also includes a term for the total space used in all bins, rewarding solutions that aim to fill up bins as much as possible.
  
  The score function combines these terms with the aim of minimizing the total number of bins used and also ensuring a uniform distribution of items among bins.
  """
  remaining_capacity = bins - item
  total_space_used = np.sum(bins)
  avg_remaining_capacity = np.mean(remaining_capacity)
  var_remaining_capacity = np.var(remaining_capacity)
  
  score = np.log(1 / (remaining_capacity + 1)) * np.exp(-(total_space_used - avg_remaining_capacity)**2 / (2 * var_remaining_capacity + 1e-5))
  return score