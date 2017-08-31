import os
from germannewssites.datasets.load_utils import load_json_dataset

train_path = 'germannewssites/datasets/data/train'
test_path = 'germannewssites/datasets/data/test'

mapping = {'1k_3': {'path': os.path.join(train_path, '1k_3')},
           '1k_5': {'path': os.path.join(train_path, '1k_5')},
           '2k_3': {'path': os.path.join(train_path, '2k_3')},
           '2k_5': {'path': os.path.join(train_path, '2k_5')},
           '4k_3': {'path': os.path.join(train_path, '4k_3')},
           '4k_5': {'path': os.path.join(train_path, '4k_5')},
           '8k_3': {'path': os.path.join(train_path, '8k_3')},
           '8k_5': {'path': os.path.join(train_path, '8k_5')},
           '16k_3': {'path': os.path.join(train_path, '16k_3')},
           '16k_5': {'path': os.path.join(train_path, '16k_5')},
           }
test_mapping = {3: {'path': os.path.join(test_path, '3_sites')},
           5: {'path': os.path.join(test_path, '5_sites')},
           }

def load(site='16k_5'):
    X, y = load_json_dataset(mapping[site]["path"])
    return X, y

def load_test(site='5'):
    X, y = load_json_dataset(test_mapping[site]["path"])
    return X, y
