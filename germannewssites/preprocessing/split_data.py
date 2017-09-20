import json
import sys
import spacy
import os
import logging
import codecs
import random

base_path = '../datasets/data'
faz_path = os.path.join(base_path, 'faz_nlp.json')
sueddeutsche_path = os.path.join(base_path, 'sueddeutsche_nlp.json')
zeit_path = os.path.join(base_path, 'zeit_nlp.json')
spiegel_path = os.path.join(base_path, 'spiegel_nlp.json')
welt_path = os.path.join(base_path, 'welt_nlp.json')

save_train = '../datasets/data/train/16k_5'
save_test = '../datasets/data/test/5_sites'

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
    values = [v for k,v in items.items()]
    print ("Elements in File: {}".format(len(values)))

    for i in range(20000):
        try:
            x = random.choice(values)
        except IndexError:
            print("nothing found")
        if i >= 16000:
            docs_test.append(random.choice(values))
            continue
        docs_train.append(random.choice(values))

    print ("Elements in Test: {}".format(len(docs_test)))
    print ("Elements in Train: {}".format(len(docs_train)))

    with codecs.open(os.path.join(save_train, filename + "_train.json"), 'w', encoding='utf-8') as file:
        json.dump(docs_train, file, ensure_ascii=False)

    with codecs.open(os.path.join(save_test, filename + "_test.json"), 'w', encoding='utf-8') as file:
        json.dump(docs_test, file, ensure_ascii=False)

''' splits training and test data from natural language processed data '''
if __name__ == "__main__":
    file_name = mapping[sys.argv[1]]
    comment_items = read_file_to_json(file_name)

    extract_and_save(comment_items, sys.argv[1])
