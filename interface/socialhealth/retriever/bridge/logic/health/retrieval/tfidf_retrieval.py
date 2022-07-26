import codecs
import itertools
import numpy as np

from .requires import *


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
        except:
            pass
    return vector


def search(query, k):
    scores = []
    tokens = process_query(query)
    query_vector = find_query_vector(tokens)

    for doc in doc_term_mat.A:
        scores.append(np.dot(query_vector, doc) /
                      (np.linalg.norm(query_vector) * np.linalg.norm(doc)))
    return np.array(scores).argsort()[-k:][::-1]


def main(query):
    print("training tfidf retrieval system for health")

    x = search(query, k=50)
    result = set()
    for index in x:
        # temp = {'title': bioset[index]['title'], 'link': bioset[index]['link']}
        temp = '' + bioset[index]['title'] + '\n' + bioset[index]['link']

        # print(bioset[index]['title'])
        # print(bioset[index]['link'])
        # print()
        result.add(temp)

    print("training tfidf retrieval system for health done")

    return list(result)


def retrieve(search_term):
    return main(search_term)
