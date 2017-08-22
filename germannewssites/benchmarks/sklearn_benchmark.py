from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn import metrics
import logging


logger = logging.getLogger(__name__)

class SklearnBenchmark():

    def __init__(self, n_folds=5):
        self.n_folds = n_folds

    def run(self, X_train, y_train, X_test, y_test, profiler):
        skf = StratifiedKFold(y_train, n_folds=self.n_folds,
                              shuffle=True, random_state=123)
        fold = 1
        for train_index, test_index in skf:
            X_train_fold, y_train_fold = [X_train[i] for i in train_index], [y_train[i] for i in train_index]
            X_test_fold, y_test_fold = [X_train[i] for i in test_index], [y_train[i] for i in test_index]
            logger.info('Training on {} instances!'.format(len(train_index)))
            profiler.train(X_train_fold, y_train_fold)
            logger.info('Testing on fold {} with {} instances'.format(
                fold, len(test_index)))
            y_pred_fold = profiler.predict(X_test_fold)
            metrics.classification_report(y_test_fold, y_pred_fold)
            fold = fold + 1


        ''' Testing grid search with basic Features '''
        vectorizer = TfidfVectorizer(max_df=0.95, min_df=3)
        X_train_grid = vectorizer.fit_transform(X_train)

        tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                             'C': [1, 10, 100, 1000]},
                            {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

        scores = ['precision', 'recall']
        for score in scores:
            print("# Tuning hyper-parameters for %s" % score)
            print()

            clf = GridSearchCV(SVC(), tuned_parameters, n_jobs=-1)
            clf.fit(X_train_grid, y_train)

            print("Best parameters set found on development set:")
            print()
            print(clf.best_params_)
            print()
            print("Grid scores on development set:")
            print()
            means = clf.cv_results_['mean_test_score']
            stds = clf.cv_results_['std_test_score']
            for mean, std, params in zip(means, stds, clf.cv_results_['params']):
                print("%0.3f (+/-%0.03f) for %r"
                      % (mean, std * 2, params))
            print()

        '''
        if X_test:
            logger.info('Training on {} instances!'.format(len(X_train)))
            profiler.train(X_train, y_train)
            logger.info('Testing on {} instances!'.format(len(X_test)))
            y_pred = profiler.predict(X_test)

            # Print the classification report
            print(metrics.classification_report(y_test, y_predicted,
                                                target_names=dataset.target_names))

            # Print and plot the confusion matrix
            cm = metrics.confusion_matrix(y_test, y_predicted)
            print(cm)

            # import matplotlib.pyplot as plt
            # plt.matshow(cm)
            # plt.show()
        '''
