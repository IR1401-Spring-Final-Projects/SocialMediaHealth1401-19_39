import numpy as np
import fasttext.util
from .requires import *


def fasttext(search):
    ft = fasttext.load_model('cc.fa.300.bin')
    search_words_splited = search.split(" ")
    vectors = []

    search_words = []
    for x in search_words_splited:
        if x not in stopwords:
            search_words.append([normalizer.normalize(lemmatizer.lemmatize(x))])

    for word in search_words:
        vectors.append(np.array(ft.get_word_vector(word[0])))

    vectors = np.array(vectors)
    search_vector = np.mean(vectors, axis=0)

    doc_vectors = []
    for i in range(len(all_tokens_nonstop)):
        vectors = []
        for word in all_tokens_nonstop[i]:
            vectors.append(np.array(ft.get_word_vector(word)))
        vectors = np.array(vectors)
        doc_vectors.append(np.mean(vectors, axis=0))

    distances_list = []
    for doc_vector in doc_vectors:
        distances_list.append(
            np.dot(search_vector, doc_vector) / (np.linalg.norm(search_vector) * np.linalg.norm(doc_vector)))

    query(distances_list, 10)


def query(distance_list, k=10):
    sorted_dist_list = []
    for doc_vec in distance_list:
        sorted_dist_list.append(doc_vec)
    sorted_dist_list.sort(reverse=True)
    doc_indices = []
    for i in range(k):
        doc_indices.append(distance_list.index(sorted_dist_list[i]))

    # print(doc_indices)
    for i in range(k):
        print(bioset[doc_indices[i]]['title'])
        print(bioset[doc_indices[i]]['link'])
        print()


def retrieve(search_term):
    return fasttext(search_term)