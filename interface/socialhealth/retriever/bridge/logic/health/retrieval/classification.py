import numpy as np
import sklearn as sk
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, precision_score, recall_score

from .requires import *


def classification():
    print("training classification retrieval system for health")
    true_labels = true_labels_func()
    print("training classification retrieval system for health done")


def true_labels_func():
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


def regression_model(true_labels):
    TEST_SIZE = 0.2
    X_train, X_test, y_train, y_test = sk.model_selection.train_test_split(doc_term_mat, true_labels,
                                                                           test_size=TEST_SIZE)
    logistic_regression = sk.linear_model.LogisticRegression()
    logistic_regression.fit(X_train, y_train)

    y_pred = logistic_regression.predict(X_test)

    print("logistic regression score", logistic_regression.score(X_test, y_test))
    print("confusion matrix: \n", confusion_matrix(y_test, y_pred))
    print("accuracy score: ", accuracy_score(y_test, y_pred))
    print("f1 score: ", f1_score(y_test, y_pred, average="macro"))
    print("precision score: ", precision_score(y_test, y_pred, average="macro"))
    print("recall score: ", recall_score(y_test, y_pred, average="macro"))
