import random

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