from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier
from ..tokenizer.token_filter import TokenFilter

import os, errno
import codecs
import logging

logger = logging.getLogger(__name__)

class GridSearchBenchmark():

    def __init__(self, n_folds=5):
        self.n_folds = n_folds

    def run(self, X_train, y_train, profiler, parameters, output_folder_name):
        directory_name = "results_grid_search/{}".format(output_folder_name)

        logger.info('Check if Grid search was done / creating Folder...')
        try:
            os.makedirs(directory_name)
        except OSError as e:
            if e.errno != errno.EEXIST:
                print("Grid search already done!")
                raise

        print("Testing grid search with following parametes")
        for parameter in parameters:
            print(parameter, parameters[parameter])

        result = []
        score = 'f1_macro'
        print("# Tuning hyper-parameters for %s" % score)
        print()
        result.append("# Tuning hyper-parameters for %s" % score)

        # debug level verbosity the higher the more prints
        grid_search = GridSearchCV(profiler.pipeline, parameters, n_jobs=-1, verbose=10, scoring=score)
        grid_search.fit(X_train, y_train)

        print("Best parameters set found on development set:")
        print()
        print(grid_search.best_params_)
        result.append("Best parameters set found on development set:")
        result.append(str(grid_search.best_params_))

        print()
        print("Grid scores on development set:")
        print()
        result.append("Grid scores on development set:")
        means = grid_search.cv_results_['mean_test_score']
        stds = grid_search.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, grid_search.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean, std * 2, params))
            result.append("%0.3f (+/-%0.03f) for %r"
                  % (mean, std * 2, params))
        print()

        with codecs.open(os.path.join(directory_name + "/grid_search_result.txt"), 'w') as file:
            for line in result:
                file.write(line + '\n')
            file.close()
