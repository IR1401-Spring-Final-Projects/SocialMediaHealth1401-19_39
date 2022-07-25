
import numpy as np
import json
import pandas as pd
import nltk
from hazm import *
import itertools
from sklearn.feature_extraction.text import TfidfVectorizer
import codecs


vectorizer = TfidfVectorizer(use_idf = True, norm ='l2', ngram_range=(1,2), analyzer='word')
doc_term_mat = vectorizer.fit_transform([' '.join(doc) for doc in all_tokens_nonstop_lemstem])
vocabulary = vectorizer.get_feature_names_out()


normalizer = Normalizer()

stop_words = [normalizer.normalize(x.strip()) for x in codecs.open('stopwords.txt', 'r', 'utf-8').readlines()]

stemmer = Stemmer()
lemmatizer = Lemmatizer()

f = open('bio.json')
bioset = json.load(f)
f.close()


def process_query(query):
    splitted_input = query.split(" ")
    nsi = []
    stopwords = [normalizer.normalize(x.strip()) for x in codecs.open('stopwords.txt','r','utf-8').readlines()]
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


def search(query, k):
    scores = []
    tokens = process_query(query)
    query_vector = find_query_vector(tokens)

    for doc in doc_term_mat.A:
        scores.append(np.dot(query_vector, doc)/(np.linalg.norm(query_vector)*np.linalg.norm(doc)))
    return np.array(scores).argsort()[-k:][::-1]

def main(query):
    x = search(query, k = 10)
    result = 10
    for index in x:
        print(bioset[index]['title'])
        print(bioset[index]['link'])
        print()
        result.append()
