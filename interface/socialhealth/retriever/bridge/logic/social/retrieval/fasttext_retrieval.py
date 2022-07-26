import numpy as np
import pandas as pd
from gensim.models import FastText
from training import Query, RetrievalSystemBase
import training


class FastTextRetrieval(RetrievalSystemBase):
    def __init__(self, k=10):
        self.k = k

    def train(self, df: pd.DataFrame):
        self.model = FastText(
            sentences=df['text_preprocessed'].apply(lambda x: x.split()).tolist(),
            sg=1,
            vector_size=110,
            epochs=10,
        )
        self.document_vectors = np.ndarray(shape=(len(df), 110))
        self.document_text_by_idx = {}
        for doc_idx, (doc, text) in enumerate(zip(df['text_preprocessed'], df['text'])):
            self.document_text_by_idx[doc_idx] = text
            splitted = doc.split()
            if len(splitted) == 0:
                continue
            document_vector = np.mean(self.model.wv[splitted], axis=0)
            self.document_vectors[doc_idx] = document_vector

    def retrieve(self, query: Query) -> list:
        query_vector = np.mean(self.model.wv[query.text.split()], axis=0)
        similarities = np.dot(self.document_vectors, query_vector)
        similarities = similarities.argsort()[-self.k:][::-1]
        return [self.document_text_by_idx[i] for i in similarities]


fasttext_retrieval = FastTextRetrieval()
fasttext_retrieval.train(training.df)


def retrieve(query):
    return fasttext_retrieval.retrieve(query)