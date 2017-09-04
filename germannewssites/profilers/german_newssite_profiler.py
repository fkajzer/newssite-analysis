from ..pipelines.pipelines import word_unigrams
from ..pipelines.pipelines import word_bigrams
from ..pipelines.pipelines import char_ngrams
from ..pipelines.pipelines import part_of_speech_features
from ..pipelines.pipelines import sentiment
from ..pipelines.pipelines import unigrams_bigrams
from sklearn.preprocessing import Normalizer
from sklearn.linear_model import Perceptron
from sklearn.pipeline import FeatureUnion
from ..utils.utils import get_classifier
from ..tokenizer.token_filter import TokenFilter
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier
#nochmal grid search machen, aber ergebnisse absaven
#test/ menge muss IMMER die gleiche sein!!!

class GermanNewssiteProfiler():
    def __init__(self, method=None, features=None):
        fs = []
        if 'unigram' in features:
            fs.append(word_unigrams())
        if 'bigram' in features:
            fs.append(word_bigrams())
        if 'bigrams' in features:
            fs.append(unigrams_bigrams())
        if 'punctuation' in features:
            fs.append(punctuation_features())
        if 'char' in features:
            fs.append(char_ngrams())
        if 'partofspeech' in features:
            fs.append(part_of_speech_features())
        if 'sentiment' in features:
            fs.append(sentiment())

        fu = FeatureUnion(fs, n_jobs=1)
        self.pipeline = Pipeline([('features', fu),
                                  ('scale', Normalizer()),
                                  ('classifier', get_classifier(method=method))])

    def train(self, X_train, Y_train):
        self.model = self.pipeline.fit(X_train, Y_train)

    def predict(self, X):
        return self.model.predict(X)
