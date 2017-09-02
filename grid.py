from germannewssites.profilers.german_newssite_profiler import GermanNewssiteProfiler
from germannewssites.benchmarks.grid_search import GridSearchBenchmark
from germannewssites.datasets.load import load

import argparse
import logging

if __name__ == '__main__':
    LOGFMT = '%(asctime)s %(name)s %(levelname)s %(message)s'
    logging.basicConfig(level=getattr(logging, 'DEBUG'), format=LOGFMT)

    argparser = argparse.ArgumentParser(description='Select which dataset to load.')
    argparser.add_argument('-d', '--data-amount', dest='data_set', type=str, default='10k',
                           help='Set data amount (1k, 5k, 10k, 20k)')
    argparser.add_argument('-s', '--sites', dest='sites', type=int, default=5,
                           help='Set number of sites (3, 5)')
    argparser.add_argument('-f', '--feature', dest='feature', type=str, default="germannewssite",
                           help='Set profiler to use (germannewssite, uni-bigram, bigrampos, partofspeech, sentiment)')
    argparser.add_argument('-c', '--classifier', dest='classifier', type=str, default="linear_svc",
                           help='Set kernel for the classifier to use (linear_svc, knn, random_forest, decision_tree)')

    logging.basicConfig(level=getattr(logging, 'DEBUG'), format=LOGFMT)
    args = argparser.parse_args()
    classifier = args.classifier
    data_set_destination = "{}_{}".format(args.data_set, args.sites)
    output_folder_name = "{}_{}_{}_{}".format(args.data_set, args.sites, args.feature, args.classifier)

    X_train, y_train = load(data_set_destination)

    if args.feature == "germannewssite":
        features = ['bigrams', 'char', 'part_of_speech'] #best feature combination
    if args.feature == "unigram":
        features = ['unigram']
    if args.feature == "uni-bigram":
        features = ['bigrams']
    if args.feature == "bigrampos":
        features = ['bigrams', 'partofspeech']
    if args.feature == "partofspeech":
        features = ['partofspeech']
    if args.feature == "sentiment":
        features = ['sentiment']

    if args.classifier == "linear_svc":
        parameters = {
            'classifier__C': (0.1, 1.0 , 100),
            'classifier__tol': (1e-2, 1e-3 , 1e-4, 1e-5, 1e-6),
            #'classifier__tol': (1e-3 , 1e-4),
        }
    if args.classifier == "svc":
        parameters = {
            #grid search has shown that only gamma affects result, C = 1.0 is default
            'classifier__C': (0.1, 1.0 , 128, 512, 8192),
            #'classifier__C': (128, 512, 2048),
            'classifier__gamma': (0.5, 0.125, 0.03125, 0.0078125)
        }
    if args.classifier == "knn":
        parameters = {
            'classifier__n_neighbors': (3,4,5,6,7,8,9,10)
        }
    if args.classifier == "random_forest":
        parameters = {
            #10 default
            'classifier__n_estimators': (10, 125, 250, 500),
            'classifier__bootstrap': (True, False),
        }
    if args.classifier == "decision_tree":
        parameters = {
            'classifier__max_depth': (None, 5, 10)
        }

    profiler_instance = GermanNewssiteProfiler(method=classifier, features=features)

    benchmark = GridSearchBenchmark()
    benchmark.run(X_train=X_train, y_train=y_train, profiler=profiler_instance, parameters=parameters, output_folder_name=output_folder_name)
