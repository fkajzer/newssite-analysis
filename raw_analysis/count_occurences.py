import json
import sys
import ijson

base_path = '../raw_analysis/'
faz_path = os.path.join(base_path, 'faz_analysis.json')
sueddeutsche_path = os.path.join(base_path, 'sueddeutsche_analysis.json')
zeit_path = os.path.join(base_path, 'zeit_analysis.json')
spiegel_path = os.path.join(base_path, 'spiegel_analysis.json')
welt_path = os.path.join(base_path, 'welt_analysis.json')

mapping = {'faz': faz_path,
           'sueddeutsche': sueddeutsche_path,
           'zeit': zeit_path,
           'spiegel': spiegel_path,
           'welt': welt_path
           }

def read_file_to_json(file_name):
    with open(file_name, 'r') as file:
        return json.load(file)

def count_words(text):
    return len(text.split(" "))

def print_nlp(sitename, items):

    result = {}
    result["comment_amount"] = len(items)

    for comment in items:
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

    with codecs.open(mapping[sitename], 'w', encoding='utf-8') as file:
        json.dump(result, file, ensure_ascii=False)

if __name__ == "__main__":
    file_name = 'raw_data/{}/{}.json'.format(sys.argv[1], sys.argv[1])
    comment_items = read_file_to_json(file_name)

    analyse(sys.argv[1], comment_items)
