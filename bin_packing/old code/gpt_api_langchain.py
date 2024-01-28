from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.chat_models import ChatOpenAI
import os

os.environ["OPENAI_API_KEY"] = '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' #Zhuohan's key
llm = ChatOpenAI(model_name = 'gpt-4', temperature = 0.75)
    
prompt_template = ChatPromptTemplate(
        messages=[SystemMessagePromptTemplate.from_template('You are an expert in combinatorial optimization and online bin-packing problem. Your are very good at python programming.You are an expert in combinatorial optimization and the online bin-packing problem. You are very good at Python programming.'),
                  HumanMessagePromptTemplate.from_template("""
Your task is to generate a better heuristic for online 1D binpacking. The requirements are as follows:
1. This heuristic serves as a score function in the <|evaluation function|>.                                                          
1. The heuristic only takes two inputs: 
    item: float, size of item to be added to the bin
    bins: numpy array, an array of capacities for each bin
2. The heuristic only returns the priority score of each bin as an array of the same size as the input `bins.`                                                                         
3. Two example heuristics are listed below as <|example heuristic 1|> and <|example heuristic 2|>. You should revise the example heuristics and propose a better one
4. Only generate one new heuristic at a time.

                                                                                                                                                                                                                                                                                                                                                                                     
<|evaluation function|>                                                      
```python
def online_binpack(
    items: tuple[float, ...], bins: np.ndarray
) -> tuple[list[list[float, ...], ...], np.ndarray]:
  \"\"\"Performs online binpacking of `items` into `bins`.\"\"\"
  # Track which items are added to each bin.
  packing = [[] for _ in bins]
  # Add items to bins.
  for item in items:
    # Extract bins that have sufficient space to fit item.
    valid_bin_indices = get_valid_bin_indices(item, bins)
    # Score each bin based on heuristic.
    priorities = priority(item, bins[valid_bin_indices])
    # Add item to bin with highest priority.
    best_bin = valid_bin_indices[np.argmax(priorities)]
    bins[best_bin] -= item
    packing[best_bin].append(item)
  # Remove unused bins from packing.
  packing = [bin_items for bin_items in packing if bin_items]
  return packing, bins  

def evaluate(instances: dict, priority) -> float:
  # Evaluate heuristic function on a set of online binpacking instances.
  # List storing number of bins used for each instance.
  num_bins = []
  # Perform online binpacking for each instance.
  for name in instances:
    instance = instances[name]
    capacity = instance['capacity']
    items = instance['items']
    # Create num_items bins so there will always be space for all items,
    # regardless of packing order. Array has shape (num_items,).
    bins = np.array([capacity for _ in range(instance['num_items'])])
    # Pack items into bins and return remaining capacity in bins_packed, which
    # has shape (num_items,).
    _, bins_packed = online_binpack(items, bins, priority)
    # If remaining capacity in a bin is equal to initial capacity, then it is
    # unused. Count number of used bins.
    num_bins.append((bins_packed != capacity).sum())
  # Score of heuristic function is negative of average number of bins used
  # across instances (as we want to minimize number of bins).
  return -np.mean(num_bins)                                                                                                                                       
```

                                                                                                                                       
<|example heuristic|>
```python 
{code1}
```

<|example heuristic 2|>
```python                                                                                                                                                                                                                                
{code2}
``` 
                                                                                                                                                            
<|generated heuristic|>                                                                        
""")],input_variables=["code1", "code2"])

code1 = "def priority(item: float, bins: np.ndarray) -> np.ndarray:\n  def s(bin, item):\n    if bin - item <= 2:\n      return 4\n    elif (bin - item) <= 3:\n      return 3\n    elif (bin - item) <= 5:\n      return 2\n    elif (bin - item) <= 7:\n      return 1\n    elif (bin - item) <= 9:\n      return 0.9\n    elif (bin - item) <= 12:\n      return 0.95\n    elif (bin - item) <= 15:\n      return 0.97\n    elif (bin - item) <= 18:\n      return 0.98\n    elif (bin - item) <= 20:\n      return 0.98\n    elif (bin - item) <= 21:\n      return 0.98\n    else:\n      return 0.99\n\n  return np.array([s(b, item) for b in bins])" 
     
code2 =  "def priority(item: float, bins: np.ndarray) -> np.ndarray:\n    return -(bins-item)"

prompt_message = prompt_template.format_prompt(code1=code1, code2=code2)

answer = llm(prompt_message.to_messages()).content

print(answer)
     