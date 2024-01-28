import os
import json
import random
import argparse
import importlib
import numpy as np

from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.callbacks import get_openai_callback

# arguments
parser = argparse.ArgumentParser(description='Bin-packing Agent')

## for llm
parser.add_argument('--model', default='gpt-4', type=str, help='llm name')
parser.add_argument('--temperature', default=0.75, type=float, help='temperature')
## for design
parser.add_argument('--iter_num', default=10, type=int, help='number of iterations')
parser.add_argument('--iter_type', default='random', choices=['random', 'best', 'topk'], type=str, help='type of iteration')
parser.add_argument('--iter_topk', default=None, type=int, help='if using topk, the number of best to choose from')
## API key
os.environ["OPENAI_API_KEY"] = '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' #Zhuohan's key


def main():
    args = parser.parse_args()

    llm = ChatOpenAI(model_name = args.model, temperature = args.temperature)
    
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

                                                                                                                                       
<|example heuristic 1|>
```python 
{code1}
```

<|example heuristic 2|>
```python                                                                                                                                                                                                                                
{code2}
``` 
                                                                                                                                                            
<|generated heuristic|>                                                                        
""")],input_variables=["code1", "code2"])
    
    # initialize database
    with open('database.json', 'r') as f:
        database = f.read()
        database = json.loads(database)
    
    # load test data
    with open('test_dataset.json', 'r') as f:
        test_data = f.read()
        test_data = json.loads(test_data)

    # start designing
    for iter in range(args.iter_num):

        print(f'-------- [ITERATION [{iter}]] ---------')


        # pick example heuristic
        if args.iter_type == 'random':
            key1, key2 = random.sample(list(database.keys()), 2)   
        elif args.iter_type == 'best':
            score_to_iteration = {v['score']: k for k, v in database.items()}
            top_2_scores = sorted(score_to_iteration, reverse=True)[:2]
            key1, key2 = [score_to_iteration[score] for score in top_2_scores]
        elif args.iter_type == 'topk':
            score_to_iteration = {v['score']: k for k, v in database.items()}
            if len(list(database.keys())) >= args.iter_topk:
                top_k_scores = sorted(score_to_iteration, reverse=True)[:args.iter_topk]
                sample_2_scores = random.sample(top_k_scores, 2) 
                key1, key2 = [score_to_iteration[score] for score in sample_2_scores]
            else:
                key1, key2 = random.sample(list(database.keys()), 2)  
        else:
           print('!!! Iteration type does not exist. Break !!!')
           break

        code1 = database[key1]['code']
        code2 = database[key2]['code']
        parents_code = [key1, key2]


        # format prompt
        prompt_message = prompt_template.format_prompt(code1=code1, code2=code2)

        # generate
        with get_openai_callback() as cb:
            answer = llm(prompt_message.to_messages()).content
            print(f'[ANSWER] \n {answer} \n\n')

            # parse answer
            generated_code = parse_answer(answer)
            # record token/api cost
            print(f'[COST] \n {cb} \n\n')
            
        if generated_code!= None:
            database = update_function(generated_code, test_data, database, iter, evaluate, parents_code)
            if database != None:
              with open('record_database.json', 'w') as j:
                  j.write(json.dumps(database, indent=4))
            else:
               continue
        else:
            continue

        


def parse_answer(answer):
    '''
    input:
        answer: model answer, String
    
    output:
        response: 
            - code, String
            - return None if the the format of answer is wrong, which will trigger regeneration
    '''

    answer_split = answer.split('\n')
    if '```python' in answer_split[0] and '```' in answer_split[-1] and 'python' not in answer_split[-1]:
        generated_code = '\n'.join(answer_split[1:-1])
        #intuition = get_intuition(answer)
        return generated_code
    else:
        idx1 = -1
        idx2 = -1
        for i in range(len(answer_split)):
            if '```python' in answer_split[i]:
                idx1 = i
            elif '```' in answer_split[i] and 'python' not in answer_split[i]:
                idx2 = i
        if -1 not in [idx1, idx2]:
            generated_code = '\n'.join(answer_split[idx1+1:idx2])
            return generated_code
        else:
            print('!!! Wrong format. Cannot parse answer. Regenerate !!!')
            return None



def update_function(code: str, instances: dict, database:dict, num_iter: int, eval_fun, parents_code: list) -> dict:
    
    # call new heuristic
    
    module_name = f"my_function_{num_iter}"
    with open(f'./cody/{module_name}.py', 'w') as file:
        file.write('import numpy as np \n' + code)
    module = importlib.import_module('cody.'+module_name)
    priority = getattr(module, 'priority')

    # compute score
    opt_num_bins = get_opt_num_bins(instances)

    try:
      avg_num_bins = -eval_fun(instances, priority)
      score = (avg_num_bins - opt_num_bins) / opt_num_bins
      score_to_iteration = {v['score']: k for k, v in database.items()}
      print(f'[SCORE] \n {score} \n\n')
      print(score_to_iteration)

      # record new heuristic
      if score in score_to_iteration.keys():
          print(f'!!! Duplicated score {score}. Equivalent heuritic to {score_to_iteration[score]} !!!')
          database[f"iter {num_iter}"] = {"code": code, 
                                          "score": score,
                                          "parents": parents_code,
                                          "equivalency": score_to_iteration[score]}
      else:
          database[f"iter {num_iter}"] = {"code": code, 
                                          "score": score,
                                          "parents": parents_code,
                                          "equivalency": "None"}
      return database
    except:
      print('!!! Error in evaluating the heuristic. Break !!!')
      return None
      
    


def l1_bound(items: tuple[int, ...], capacity: int) -> float:
  """Computes L1 lower bound on OPT for bin packing.

  Args:
    items: Tuple of items to pack into bins.
    capacity: Capacity of bins.

  Returns:
    Lower bound on number of bins required to pack items.
  """
  return np.ceil(np.sum(items) / capacity)


def l1_bound_dataset(instances: dict) -> float:
  """Computes the mean L1 lower bound across a dataset of bin packing instances.

  Args:
    instances: Dictionary containing a set of bin packing instances.

  Returns:
    Average L1 lower bound on number of bins required to pack items.
  """
  l1_bounds = []
  for name in instances:
    instance = instances[name]
    l1_bounds.append(l1_bound(instance['items'], instance['capacity']))
  return np.mean(l1_bounds)

def get_opt_num_bins(instance):
    return l1_bound_dataset(instance)

import numpy as np
def get_valid_bin_indices(item: float, bins: np.ndarray) -> np.ndarray:
  """Returns indices of bins in which item can fit."""
  return np.nonzero((bins - item) >= 0)[0]


def online_binpack(
    items: tuple[float, ...], bins: np.ndarray
, priority) -> tuple[list[list[float, ...], ...], np.ndarray]:
  """Performs online binpacking of `items` into `bins`."""
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


# @funsearch.run
def evaluate(instances: dict, priority) -> float:
  """Evaluate heuristic function on a set of online binpacking instances."""
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

if __name__ == '__main__':
    main()