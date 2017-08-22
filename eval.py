from germannewssites.profilers.german_newssite_profiler import GermanNewssiteProfiler
from germannewssites.benchmarks.sklearn_benchmark import SklearnBenchmark
from germannewssites.config import Config
from germannewssites.datasets.load import load
import argparse
import logging

def configure(conf):

    @conf.profiler('german-newssite-profiler', method='logistic_regression')
    def build_newssite_profiler(**args):
        features = ['unigram', 'bigram', 'punctuation', 'char']
        return GermanNewssitesProfiler(method='logistic_regression', features=features)

    @conf.dataset('german-newssite-data')
    def build_dataset(site=None):
        X, y = load()
        return X, y

def pretty_list(items):
    return ', '.join([x for x in items])

if __name__ == '__main__':
    '''
    conf = Config()
    argparser = argparse.ArgumentParser(description='Author Profiling Evaluation')
    argparser.add_argument('-l', '--log-level', dest='log_level', type=str, default='INFO',
                           help='Set log level (DEBUG, INFO, ERROR)')

    argparser.add_argument('-c', '--train_corpus', dest='training_corpus', type=str, required=True,
                           help='Set name of the training corpus used for the evaluation: ' + pretty_list(
                               conf.get_dataset_names()))

    argparser.add_argument('-t', '--test_corpus', dest='test_corpus', type=str, required=False,
                           help='Set name of the test corpus used for the evaluation: ' + pretty_list(
                               conf.get_dataset_names()))

    argparser.add_argument('-p', '--profiler', dest='profiler_name', type=str, required=True,
                           help='Name of the invoked profiler: ' + pretty_list(conf.get_profiler_names()))

    args = argparser.parse_args()
    '''
    LOGFMT = '%(asctime)s %(name)s %(levelname)s %(message)s'

    '''
    logging.basicConfig(level=getattr(logging, args.log_level), format=LOGFMT)

    X_train, y_train = conf.get_dataset(
        args.training_corpus)
    if args.test_corpus:
        X_test, y_test = conf.get_dataset(args.test_corpus)
    else:
        X_test, y_test = None
    '''
    logging.basicConfig(level=getattr(logging, 'DEBUG'), format=LOGFMT)

    X_train, y_train = load()
    features = ['unigram', 'bigram', 'punctuation', 'char']
    profiler_instance = GermanNewssiteProfiler(method='logistic_regression', features=features)

    benchmark = SklearnBenchmark()
    benchmark.run(X_train=X_train, y_train=y_train, X_test=None, y_test=None, profiler=profiler_instance)

    #benchmark.run(X_train=X_train, y_train=y_train,
    #              X_test=X_test, y_test=y_test,
    #              profiler=profiler_instance)
