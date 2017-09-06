from germannewssites.profilers.german_newssite_profiler import GermanNewssiteProfiler
from germannewssites.benchmarks.sklearn_benchmark import SklearnBenchmark
from germannewssites.datasets.load import load, load_test

import argparse
import logging

if __name__ == '__main__':
    LOGFMT = '%(asctime)s %(name)s %(levelname)s %(message)s'


    argparser = argparse.ArgumentParser(description='Select which dataset to load.')
    argparser.add_argument('-d', '--data-set', dest='data_set', type=str, default='8k',
                           help='Set data amount (1k, 2k, 4k, 8k, 16k)')
    argparser.add_argument('-s', '--sites', dest='sites', type=int, default=5,
                           help='Set number of sites (3, 5)')
    argparser.add_argument('-c', '--classifier', dest='classifier', type=str, default="linear_svc",
                           help='Set the classifier to use (linear_svc, knn, random_forest)')

    logging.basicConfig(level=getattr(logging, 'DEBUG'), format=LOGFMT)
    args = argparser.parse_args()
    data_set_destination = "{}_{}".format(args.data_set, args.sites)
    classifier = args.classifier

    X_train, y_train = load(data_set_destination)
    X_test, y_test = load_test(args.sites)

    features = ['unigram', 'bigram', 'uni-bigram', 'char', 'partofspeech', 'germannewssite']
    #features = ['partofspeech']

    if args.classifier == "linear_svc":
        hyper_parameters = {
            'classifier__C': (0.001, 0.01, 0.1, 1.0 , 10, 100, 1000),
        }
    if args.classifier == "svc":
        hyper_parameters = {
            'classifier__C': (0.001, 1.0, 100),
            'classifier__gamma': (0.001, 0.01, 0.1, 1.0, 10)
        }
    if args.classifier == "knn":
        hyper_parameters = {
            'classifier__n_neighbors': (1,2,3,4,5,6,7,8,9,10,11,12,13,14,15)
        }
    if args.classifier == "random_forest":
        hyper_parameters = {
            "classifier__n_estimators": (10, 20, 50),
            "classifier__max_depth": [10, 50, 100, None],
            "classifier__min_samples_split": [2, 3, 10],
            "classifier__min_samples_leaf": [1, 3, 10],
        }

    #run benchmark for each feature
    benchmark = SklearnBenchmark()
    for feature in features:
        output_folder_name = "{}/{}/{}/{}".format(args.data_set, args.sites, args.classifier, feature)

        profiler_instance = GermanNewssiteProfiler(method=classifier, feature=feature)
        benchmark.run(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, profiler=profiler_instance, output_folder_name=output_folder_name, hyper_parameters=hyper_parameters)
