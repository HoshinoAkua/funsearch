import random
import json

def generate(item_num):
    return [random.randint(20,100) for _ in range(item_num)]

        
def test_dataset_generataion(instance_num ,item_num, bin_capacity):
    instances = {}
    for i in range(instance_num):
        instance = {}
        instance['capacity'] = bin_capacity
        instance['num_items'] = item_num
        instance['items'] = generate(item_num)
        instances[f'u{item_num}_{i}'] = instance

    return instances

test_data = test_dataset_generataion(20, 120, 150)
valid_data = test_dataset_generataion(20, 250, 150)

with open('test_dataset.json', 'w') as j:
   j.write(json.dumps(test_data))

with open('valid_dataset.json', 'w') as j:
   j.write(json.dumps(valid_data))