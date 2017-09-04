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
    argparser.add_argument('-f', '--features', dest='feature', type=str, default="germannewssite",
                           help='Set profiler to use (germannewssite, unigram, bigram, bigrampos, partofspeech)')
    argparser.add_argument('-c', '--classifier', dest='classifier', type=str, default="linear_svc",
                           help='Set kernel for the classifier to use (linear_svc, knn, random_forest, decision_tree)')

    logging.basicConfig(level=getattr(logging, 'DEBUG'), format=LOGFMT)
    args = argparser.parse_args()
    data_set_destination = "{}_{}".format(args.data_set, args.sites)
    classifier = args.classifier
    output_folder_name = "{}_{}_{}_{}".format(args.data_set, args.sites, args.feature, args.classifier)

    X_train, y_train = load(data_set_destination)
    X_test, y_test = load_test(args.sites)

    #features = ['unigram', 'bigram', 'bigrams', 'punctuation', 'char', 'part_of_speech', 'sentiment']
    if args.feature == "germannewssite":
        features = ['bigrams', 'char', 'partofspeech'] #best feature combination
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

    profiler_instance = GermanNewssiteProfiler(method=classifier, features=features)

    benchmark = SklearnBenchmark()
    benchmark.run(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, profiler=profiler_instance, output_folder_name=output_folder_name)
