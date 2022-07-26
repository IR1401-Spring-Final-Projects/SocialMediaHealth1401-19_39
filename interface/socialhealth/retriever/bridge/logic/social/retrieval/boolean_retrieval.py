import pandas as pd
import functools
import numpy as np
from retriever.bridge.logic.social.retrieval.training import Query, RetrievalSystemBase, df


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
        try: 
            idx = self.word_to_idx[word]
        except KeyError:
            return []
        return [self.idx_to_document[i] for i in np.where(self.document_word_matrix[:, idx] == 1)[0]]


print("training boolean retrieval system")
boolean_retrieval_system = BooleanRetrieval()
boolean_retrieval_system.train(df)
print("training boolean retrieval system done")

def retrieve(query):
    print(query)
    results = boolean_retrieval_system.retrieve(Query(query))
    print(boolean_retrieval_system.retrieve(Query("happy")))
    print(len(results))
    return results
