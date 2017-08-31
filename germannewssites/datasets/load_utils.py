import logging
import warnings
import os
import json

warnings.filterwarnings('error')
logger = logging.getLogger(__name__)

def read_file_to_json(file_name):
    with open(file_name, 'r') as file:
        return json.load(file)

def parse_json(filename):
    """parses json files, that contains all comments from a site"""
    logger.debug(u'Parse {}'.format(filename))
    return read_file_to_json(filename)

def parse_json_files(files_dir):
    import multiprocessing
    logger.info('Reading json files from: {}'.format(files_dir))

    filenames = sorted([os.path.join(files_dir, f)
                        for f in os.listdir(files_dir) if f.endswith('.json')])
    logger.info('Files: {}'.format(len(filenames)))

    cpu_count = multiprocessing.cpu_count()
    n_jobs = cpu_count - 1 if cpu_count >= 2 else 1
    logger.info('Parallel jobs: {}'.format(n_jobs))

    docs = []
    total = len(filenames)
    if n_jobs == 1:
        for filename in filenames:
            documents = parse_json(filename)
            for doc in documents:
                docs.append(doc)
    else:
        pool = multiprocessing.Pool(n_jobs)
        files = pool.map(parse_json, filenames)
        for file in files:
            for doc in file:
                docs.append(doc)

    logger.info('Parsed {} json files in {}'.format(len(docs), files_dir))
    return docs

def load_json_dataset(files_dir):
    X = parse_json_files(files_dir)
    y = []

    for doc in X:
        y.append(doc["target"])
        del doc["target"]
        #text_len = len(concat_sentences(doc))
        #logger.info(u'chars={chars}'.format(chars=text_len))

    return X, y

def concat_sentences(doc):
    return [(doc[sentence]['sentence']) for sentence in doc]
