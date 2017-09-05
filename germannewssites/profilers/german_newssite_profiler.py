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

class GermanNewssiteProfiler():
    def __init__(self, method=None, feature=None):
        fs = []
        if 'unigram' == feature:
            fs.append(word_unigrams())
        if 'bigram' == feature:
            fs.append(word_bigrams())
        if 'uni-bigram' == feature:
            fs.append(unigrams_bigrams())
        if 'char' == feature:
            fs.append(char_ngrams())
        if 'partofspeech' == feature:
            fs.append(part_of_speech_features())
        if 'sentiment' == feature:
            fs.append(sentiment())
        if 'germannewssite' == feature:
            fs.append(unigrams_bigrams())
            fs.append(char_ngrams())
            fs.append(part_of_speech_features())

        fu = FeatureUnion(fs, n_jobs=1)
        self.pipeline = Pipeline([('features', fu),
                                  ('scale', Normalizer()),
                                  ('classifier', get_classifier(method=method))])

    def train(self, X_train, Y_train):
        self.model = self.pipeline.fit(X_train, Y_train)

    def predict(self, X):
        return self.model.predict(X)

    def set_params(self, params):
        self.pipeline.set_params(**params)
