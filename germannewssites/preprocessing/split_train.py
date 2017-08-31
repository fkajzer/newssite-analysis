import json
import sys
import spacy
import os
import logging
import codecs
import random

base_path = '../datasets/data/train/16k_5'
faz_path = os.path.join(base_path, 'faz_train.json')
sueddeutsche_path = os.path.join(base_path, 'sueddeutsche_train.json')
zeit_path = os.path.join(base_path, 'zeit_train.json')
spiegel_path = os.path.join(base_path, 'spiegel_train.json')
welt_path = os.path.join(base_path, 'welt_train.json')

save_train = '../datasets/data/train/1k_5'

mapping = {'faz': faz_path,
           'sueddeutsche': sueddeutsche_path,
           'zeit': zeit_path,
           'spiegel': spiegel_path,
           'welt': welt_path
           }

logger = logging.getLogger(__name__)

def read_file_to_json(file_name):
    with open(file_name, 'r') as file:
        return json.load(file)

def extract_and_save(items, filename):

    docs_train = []
    docs_test = []
    print ("Elements in File: {}".format(len(items)))

    for i in range(1000):
        try:
            x = random.choice(items)
        except IndexError:
            print("nothing found")
        docs_train.append(random.choice(items))

    print ("Elements in Train: {}".format(len(docs_train)))

    with codecs.open(os.path.join(save_train, filename + "_train.json"), 'w', encoding='utf-8') as file:
        json.dump(docs_train, file, ensure_ascii=False)

if __name__ == "__main__":
    file_name = mapping[sys.argv[1]]
    comment_items = read_file_to_json(file_name)

    extract_and_save(comment_items, sys.argv[1])
