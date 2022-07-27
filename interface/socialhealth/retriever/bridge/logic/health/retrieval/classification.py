import itertools

import numpy as np
import sklearn as sk
from scipy import sparse
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, precision_score, recall_score

from .requires import *

dic = {}
true_labels = []


def classification():
    global true_labels
    true_labels = true_labels_func()


def get_category(index):
    return dict(zip(dic.values(), dic.keys()))[index]


def true_labels_func():
    global dic

    categories = set()
    for i in bioset:
        categories.add(i['categories'][1])
    dic = {}
    counter = 0
    for category in categories:
        dic[category] = counter
        counter += 1
    true_labels = []
    for i in bioset:
        category = i['categories'][1]
        true_labels.append(dic[category])
    return np.array(true_labels)


def process_query(query):
    splitted_input = query.split(" ")
    nsi = []
    for x in splitted_input:
        if x not in stopwords:
            nsi.append([normalizer.normalize(lemmatizer.lemmatize(x))])
    return nsi


def find_query_vector(tokens):
    vector = np.zeros(len(vocabulary))
    for token in itertools.chain(*tokens):
        try:
            index = vectorizer.vocabulary_[token]
            vector[index] = 1
        except ValueError:
            pass
    return vector


def regression_model(query):
    global logistic_regression
    # TEST_SIZE = 0.2
    # X_train, X_test, y_train, y_test = sk.model_selection.train_test_split(doc_term_mat, true_labels,
    #                                                                        test_size=TEST_SIZE)
    # logistic_regression = sk.linear_model.LogisticRegression()
    # logistic_regression.fit(X_train, y_train)
    # print("logistic regression score", logistic_regression.score(X_test, y_test))
    # print("confusion matrix: \n", confusion_matrix(y_test, y_pred))
    # print("accuracy score: ", accuracy_score(y_test, y_pred))
    # print("f1 score: ", f1_score(y_test, y_pred, average="macro"))
    # print("precision score: ", precision_score(y_test, y_pred, average="macro"))
    # print("recall score: ", recall_score(y_test, y_pred, average="macro"))

    query_vector = find_query_vector(process_query(query))
    query_vector = sparse.csr_matrix(query_vector)
    query_class = logistic_regression.predict(query_vector)

    return get_category(query_class[0])


print("training classification retrieval system for health")
classification()
TEST_SIZE = 0.2
X_train, X_test, y_train, y_test = sk.model_selection.train_test_split(doc_term_mat, true_labels,
                                                                       test_size=TEST_SIZE)
logistic_regression = sk.linear_model.LogisticRegression()
logistic_regression.fit(X_train, y_train)
print("training classification retrieval system for health done")


def run(search_term):
    return regression_model(search_term)
