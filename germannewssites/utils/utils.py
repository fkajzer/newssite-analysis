from sklearn.linear_model import LogisticRegression

def get_classifier(method='logistic_regression'):
    if 'logistic_regression' == method:
        return LogisticRegression(C=1e3,
                                  tol=0.01,
                                  multi_class='ovr',
                                  #n_jobs=-1,
                                  solver='liblinear',
                                  random_state=123)

def extract_sentences(doc):
    return [(doc[sentence]['sentence']) for sentence in doc]
