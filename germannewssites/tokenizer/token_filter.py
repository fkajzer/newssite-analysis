import json
import sys
import os
import logging
import codecs

from ..utils.utils import extract_token, filter_token

class TokenFilter(object):
    def __init__(self, filter=None):
            self.filter = filter

    def __call__(self, doc):
        if self.filter is not None:
            return filter_token(doc, self.filter)

        return extract_token(doc)
