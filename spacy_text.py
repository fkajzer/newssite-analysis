import json
import sys
import spacy

def read_file_to_json(file_name):
    with open(file_name, 'r') as file:
        return json.load(file)

def print_nlp(nlp, items):
    comments_content_nlp = [nlp(comment['content']) for comment in items[:10]]

    docs = {}
    index = 0
    for document in comments_content_nlp:
        sentence_index = 0
        doc = {}
        for sentence in document.sents:
            sent = {}
            sentence["sentence"] = sentence
            sentence["token"] = [(word.lemma_, word.tag_, word.pos_) for word in sentence]

            doc[sentence_index] = sent
            sentence_index += 1

        docs[index] = doc
        index += 1

    print (json.dumps(docs))

if __name__ == "__main__":
    file_name = 'data/{}/{}.json'.format(sys.argv[1], sys.argv[1])
    comment_items = read_file_to_json(file_name)

    nlp = spacy.load('de')
    print_nlp(nlp, comment_items)
