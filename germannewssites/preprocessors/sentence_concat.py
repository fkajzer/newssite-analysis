from ..utils.utils import extract_sentences

class SentenceConcat(object):
    def __init__(self, only_first=False):
            self.only_first = only_first

    def __call__(self, doc):
        if self.only_first:
            return doc['0']['sentence']

        return "".join(extract_sentences(doc))
