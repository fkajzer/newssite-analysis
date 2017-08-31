from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
#benchmark mal KNN / mehrere Classifakatoren
#probier mit kernels aus

def get_classifier(method='linear'):
    if 'linear_svc' == method:
        return LinearSVC(C=1000, multi_class='ovr', random_state=123)
    if 'svc' == method:
        #rbf kernel
        return SVC(C=1000, random_state=123)
    if 'knn' == method:
        return KNN(n_jobs=-1)
    if 'random_forest' == method:
        return RandomForestClassifier(n_estimators=250,
                                      bootstrap=False,
                                      n_jobs=-1,
                                      random_state=123)
    if 'decision_tree' == method:
        return DecisionTreeClassifier(n_jobs=-1,
                                      random_state=123)

def extract_part_of_speech(doc):
    token_list = []
    for sentence in doc:
        for token in doc[sentence]['token']:
            token_list.append(token['pos'])
    return token_list

def extract_token(doc):
    token_list = []
    for sentence in doc:
        for token in doc[sentence]['token']:
            token_list.append(token['lemma'])
    return token_list

def filter_token(doc, filter):
    token_list = []
    for sentence in doc:
        for token in doc[sentence]['token']:
            if token['pos'] in filter:
                token_list.append(token['lemma'])
    return token_list
