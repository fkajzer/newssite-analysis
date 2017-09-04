from germannewssites.features.part_of_speech_features import PartOfSpeechFeatures
from germannewssites.preprocessors.sentence_concat import SentenceConcat
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer
from ..tokenizer.token_filter import TokenFilter

def part_of_speech_features():
    pipeline = Pipeline([('feature', PartOfSpeechFeatures()),
                         ('tfidf', TfidfTransformer(sublinear_tf=False)),
                         ('scale', Normalizer())])
    return ('part_of_speech_features', pipeline)

def word_unigrams():
    vectorizer = CountVectorizer(min_df=2,
                                 lowercase=False,
                                 tokenizer=TokenFilter(),
                                 ngram_range=(1, 1))
    pipeline = Pipeline([('vect', vectorizer),
                         ('tfidf', TfidfTransformer(sublinear_tf=True)),
                         ('scale', Normalizer())])
    return ('word_unigrams', pipeline)

def word_bigrams():
    pipeline = Pipeline([('vect', CountVectorizer(ngram_range=(2, 2), lowercase=False, tokenizer=TokenFilter())),
                         ('tfidf', TfidfTransformer(sublinear_tf=True)),
                         ('scale', Normalizer())])
    return ('word_bigrams', pipeline)

def unigrams_bigrams():
    pipeline = Pipeline([('vect', CountVectorizer(ngram_range=(1, 2), lowercase=False, tokenizer=TokenFilter())),
                         ('tfidf', TfidfTransformer(sublinear_tf=True)),
                         ('scale', Normalizer())])
    return ('unigrams_bigrams', pipeline)

def char_ngrams():
    vectorizer = CountVectorizer(min_df=1,
                                 preprocessor=SentenceConcat(),
                                 analyzer='char_wb',
                                 ngram_range=(4, 4))
    pipeline = Pipeline([('vect', vectorizer),
                         ('tfidf', TfidfTransformer(sublinear_tf=True)),
                         ('scale', Normalizer())])
    return ('char_ngrams', pipeline)

def sentiment():
    pipeline = Pipeline([
            ('vect', TfidfVectorizer(lowercase=False, tokenizer=TokenFilter(['ADJ', 'VERB', 'ADV']))),
            ('scale', Normalizer())])
    return ('sentiment', pipeline)
