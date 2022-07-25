import pandas as pd
import functools
import numpy as np
from training import Query, RetrievalSystemBase
import training


class BooleanRetrieval(RetrievalSystemBase):
    def __init__(self, k=10):
        self.k = k
        self.word_to_idx = None
        self.idx_to_document = None
        self.document_word_matrix = None

    def train(self, df: pd.DataFrame):
        all_words_list = df['text_preprocessed'].apply(
            lambda x: x.split()).tolist()
        all_words_list_flattened = [x for y in all_words_list for x in y]
        all_words = set(all_words_list_flattened)
        self.word_to_idx = {word: idx for idx, word in enumerate(all_words)}
        self.idx_to_document = {}
        self.document_word_matrix = np.zeros((len(df), len(all_words)))

        for doc_idx, (doc, text) in enumerate(zip(df['text_preprocessed'], df['text'])):
            self.idx_to_document[doc_idx] = text
            for word in doc.split():
                self.document_word_matrix[doc_idx, self.word_to_idx[word]] = 1

    def retrieve(self, query: Query) -> list:
        documents = [set(self.__retrieve_word(x)) for x in query.text.split()]
        intersection = functools.reduce(lambda x, y: x.intersection(y), documents)
        union = functools.reduce(lambda x, y: x.union(y), documents)
        signle = union - intersection
        length = self.k if len(intersection) > self.k else len(intersection)
        return list(intersection)[:length] + list(signle)[:self.k-length]

    def __retrieve_word(self, word: str) -> list:
        idx = self.word_to_idx[word]
        return [self.idx_to_document[i] for i in np.where(self.document_word_matrix[:, idx] == 1)[0]]


boolean_retrieval_system = BooleanRetrieval()
boolean_retrieval_system.train(training.df)


def retrieve(query):
    return boolean_retrieval_system.retrieve(query)
