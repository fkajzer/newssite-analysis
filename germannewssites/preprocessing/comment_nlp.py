import json
import sys
import spacy
import os
import logging
import codecs

base_path = '../raw_data/'
faz_path = os.path.join(base_path, 'faz/faz.json')
sueddeutsche_path = os.path.join(base_path, 'sueddeutsche/sueddeutsche.json')
zeit_path = os.path.join(base_path, 'zeit/zeit.json')
spiegel_path = os.path.join(base_path, 'spiegel/spiegel.json')
welt_path = os.path.join(base_path, 'welt/welt.json')

base_path_save = '../datasets/data/'

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

def create_token_object(token):
    return {'lemma': token.lemma_,
           'tag': token.tag_,
           'pos': token.pos_
           }

def save_nlp(nlp, items, filename):
    if filename == "faz" or filename == "zeit" or filename == "spiegel":
        comments_content_nlp = [nlp(comment['content']) for comment in items]

    if filename == "welt":
        comments_content_nlp = []
        for comment in items:
            try:
                comments_content_nlp.append(nlp(comment['contents']))
            except:
                continue

    if filename == "sueddeutsche":
        comments_content_nlp = [nlp(comment['raw_message']) for comment in items]

    docs = {}
    index = 0
    for document in comments_content_nlp:
        sentence_index = 0
        doc = {}
        doc["target"] = filename
        for sentence in document.sents:
            sent = {}
            sent["sentence"] = sentence.string
            sent["token"] = [create_token_object(token) for token in sentence]

            doc[sentence_index] = sent
            sentence_index += 1

        docs[index] = doc
        index += 1

    with codecs.open(os.path.join(base_path_save, filename + "_nlp.json"), 'w', encoding='utf-8') as file:
        json.dump(docs, file, ensure_ascii=False)

if __name__ == "__main__":
    file_name = mapping[sys.argv[1]]
    comment_items = read_file_to_json(file_name)

    nlp = spacy.load('de')
    save_nlp(nlp, comment_items, sys.argv[1])
