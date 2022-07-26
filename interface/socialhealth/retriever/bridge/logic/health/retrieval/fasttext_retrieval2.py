import numpy as np
from gensim.models import FastText
# from retriever.bridge.logic.social.retrieval.training import Query, RetrievalSystemBase
from social.retrieval.training import Query, RetrievalSystemBase

from .requires import *


class FastTextRetrieval(RetrievalSystemBase):
    def __init__(self, k=10):
        self.k = k

    def train(self, df: pd.DataFrame):
        # self.model = FastText(
        #     sentences=df['text_preprocessed'].apply(lambda x: x.split()).tolist(),
        #     sg=1,
        #     vector_size=110,
        #     epochs=10,
        # )

        self.model = FastText(
            sentences=all_tokens_nonstop_lemstem,
            sg=1,
            vector_size=110,
            epochs=10,
        )

        self.document_vectors = np.ndarray(shape=(documents_length, 110))
        self.document_text_by_idx = {}
        for doc_idx, (doc, text) in enumerate(zip(all_tokens_nonstop_lemstem, titles_links)):
            self.document_text_by_idx[doc_idx] = text
            # splitted = doc.split()
            # if len(splitted) == 0:
            #     continue
            if len(doc) == 0:
                continue
            document_vector = np.mean(self.model.wv[doc], axis=0)
            self.document_vectors[doc_idx] = document_vector

    def retrieve(self, query: Query) -> list:
        query_vector = np.mean(self.model.wv[query.text.split()], axis=0)
        similarities = np.dot(self.document_vectors, query_vector)
        similarities = similarities.argsort()[-self.k:][::-1]
        return [self.document_text_by_idx[i] for i in similarities]


print("training fasttext retrieval system")
fasttext_retrieval = FastTextRetrieval()
# fasttext_retrieval.train(df)
print("training fasttext retrieval system done")


def retrieve(query):
    return fasttext_retrieval.retrieve(Query(query))


if __name__ == '__main__':
    retrieve('تپش قلب')
