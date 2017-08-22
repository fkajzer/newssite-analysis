from germannewssites.features.punctuation_features import PunctuationFeatures
from germannewssites.preprocessors.sentence_concat import SentenceConcat
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer

def punctuation_features():
    pipeline = Pipeline([('feature', PunctuationFeatures()),
                         ('tfidf', TfidfTransformer(sublinear_tf=False)),
                         ('scale', Normalizer())])
    return ('punctuation_features', pipeline)

def word_unigrams():
    vectorizer = CountVectorizer(min_df=2,
                                 preprocessor=SentenceConcat(),
                                 ngram_range=(1, 1))
    pipeline = Pipeline([('vect', vectorizer),
                         ('tfidf', TfidfTransformer(sublinear_tf=True)),
                         ('scale', Normalizer())])
    return ('word_unigrams', pipeline)

def word_bigrams():
    pipeline = Pipeline([('vect', CountVectorizer(ngram_range=(2, 2), preprocessor=SentenceConcat())),
                         ('tfidf', TfidfTransformer(sublinear_tf=True)),
                         ('scale', Normalizer())])
    return ('word_bigrams', pipeline)

def char_ngrams():
    vectorizer = CountVectorizer(min_df=1,
                                 preprocessor=SentenceConcat(),
                                 analyzer='char_wb',
                                 ngram_range=(4, 4))
    pipeline = Pipeline([('vect', vectorizer),
                         ('tfidf', TfidfTransformer(sublinear_tf=True)),
                         ('scale', Normalizer())])
    return ('char_ngrams', pipeline)
