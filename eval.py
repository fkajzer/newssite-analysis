from germannewssites.profilers.german_newssite_profiler import GermanNewssiteProfiler
from germannewssites.benchmarks.sklearn_benchmark import SklearnBenchmark
from germannewssites.datasets.load import load, load_test

import argparse
import logging

logger = logging.getLogger(__name__)

if __name__ == '__main__':
    LOGFMT = '%(asctime)s %(name)s %(levelname)s %(message)s'


    argparser = argparse.ArgumentParser(description='Select which dataset to load.')
    argparser.add_argument('-d', '--data-set', dest='data_set', type=str, default='8k',
                           help='Set data amount (1k, 2k, 4k, 8k, 16k)')
    argparser.add_argument('-s', '--sites', dest='sites', type=int, default=5,
                           help='Set number of sites (3, 5)')
    argparser.add_argument('-c', '--classifier', dest='classifier', type=str, default="none",
                           help='Set the classifier to use (svc, linear_svc, knn, random_forest)')
    argparser.add_argument('-f', '--feature', dest='feature', type=str, default="none",
                           help='Set the feature to use (unigram, bigram, uni-bigram, char, partofspeech, germannewssite)')

    logging.basicConfig(level=getattr(logging, 'DEBUG'), format=LOGFMT)
    args = argparser.parse_args()
    data_set_destination = "{}_{}".format(args.data_set, args.sites)
    classifier = args.classifier

    X_train, y_train = load(data_set_destination)
    X_test, y_test = load_test(args.sites)

    if args.feature == "none":
        features = ['unigram', 'bigram', 'uni-bigram', 'char', 'partofspeech', 'germannewssite']
    else:
        features = [args.feature]

    if args.classifier == "none":
        classifiers = ['svc', 'linear_svc', 'random_forest', 'knn']
    else:
        classifiers = [args.classifier]

    #run benchmark for each classifier and each feature
    benchmark = SklearnBenchmark()
    for classifier in classifiers:
        if classifier == "linear_svc":
            hyper_parameters = {
                'classifier__C': (0.001, 0.01, 0.1, 1.0 , 10, 100),
            }
        if classifier == "svc":
            hyper_parameters = {
                'classifier__C': (1.0, 100, 1000),
                'classifier__gamma': (1.0, 10, 100, 1000)
            }
        if classifier == "knn":
            hyper_parameters = {
                'classifier__n_neighbors': (1,5,10,15,20,25,30,35,40,45,50,60,70,80)

            }
        if classifier == "random_forest":
            hyper_parameters = {
                #"classifier__max_depth": [10, 50, 100, None],
                "classifier__max_depth": [100, None],
                "classifier__min_samples_split": [2, 3, 10],
                "classifier__min_samples_leaf": [1, 3, 10],
            }

        for feature in features:

            logger.info("Starting Benchmark for classifier: {} with feature: {}!".format(classifier, feature))
            output_folder_name = "{}/{}/{}/{}".format(args.data_set, args.sites, classifier, feature)

            profiler_instance = GermanNewssiteProfiler(method=classifier, feature=feature)
            benchmark.run(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, profiler=profiler_instance, output_folder_name=output_folder_name, hyper_parameters=hyper_parameters)
