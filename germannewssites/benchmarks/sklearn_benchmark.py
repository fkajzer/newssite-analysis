from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn import preprocessing

import itertools
import matplotlib.pyplot as plt
import numpy as np
import logging
import os, errno
import codecs

logger = logging.getLogger(__name__)

def plot_confusion_matrix(cm, classes, filename,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting 'normalize=True'.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    plt.savefig(filename)

def grid_search(X_train, y_train, profiler, parameters, directory_name):

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

    with codecs.open(os.path.join(directory_name + "/grid_search_cv.txt"), 'w') as file:
        for line in result:
            file.write(line + '\n')
        file.close()

    return grid_search.best_params_

class SklearnBenchmark():

    def __init__(self, n_folds=5):
        self.n_folds = n_folds

    def run(self, X_train, y_train, X_test, y_test, profiler, output_folder_name, hyper_parameters):
        directory_name = "results/{}".format(output_folder_name)

        logger.info('Creating Evaluation Folder...')
        try:
            os.makedirs(directory_name)
        except OSError as e:
            if e.errno != errno.EEXIST:
                print("Evaluation already exists!")
                raise
        best_parameters = grid_search(X_train=X_train, y_train=y_train, profiler=profiler, parameters=hyper_parameters, directory_name=directory_name)

        if X_test:
            profiler.set_params(best_parameters)

            #get target names as list
            le = preprocessing.LabelEncoder()
            le.fit(y_test)
            class_names = list(le.classes_)

            logger.info('Testing...')
            logger.info('Training on {} instances!'.format(len(X_train)))
            profiler.train(X_train, y_train)
            logger.info('Testing on {} instances!'.format(len(X_test)))
            y_predicted = profiler.predict(X_test)

            #save y_predicted
            with codecs.open(os.path.join(directory_name + "/y_predicted.npy"), 'wb') as file:
                np.save(file, y_predicted)
                file.close()

            # Print the classification report and save it
            print(metrics.classification_report(y_test, y_predicted))
            print(metrics.f1_score(y_test, y_predicted, average='macro'))
            with codecs.open(os.path.join(directory_name + "/classification_report.txt"), 'w') as file:
                file.write(metrics.classification_report(y_test, y_predicted))
                file.write("Macro-F1-Score: " + str(metrics.f1_score(y_test, y_predicted, average='macro')))
                file.close()

            # Compute confusion matrix
            cnf_matrix = metrics.confusion_matrix(y_test, y_predicted)
            np.set_printoptions(precision=2)

            # Plot non-normalized confusion matrix
            plt.figure()
            plot_confusion_matrix(cnf_matrix, classes=class_names, filename=directory_name + "/cm.png",
                                  title='Confusion matrix, without normalization')

            # Plot normalized confusion matrix and save it
            plt.figure()
            plot_confusion_matrix(cnf_matrix, classes=class_names, filename=directory_name + "/cm_normalized.png", normalize=True,
                                  title='Normalized confusion matrix')

            plt.close()
