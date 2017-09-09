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
        return LinearSVC(C=1.0, multi_class='ovr', random_state=123)
    if 'svc' == method:
        return SVC(C=100, gamma=1.0, random_state=123, verbose=True)
    if 'knn' == method:
        return KNN(n_neighbors=1)
    if 'random_forest' == method:
        return RandomForestClassifier(max_depth=None, min_samples_leaf=1, min_samples_split=10, n_estimators=50,
                                    n_jobs=-1, random_state=123)

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
