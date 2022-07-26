import sklearn as sk
import pandas as pd
from retriever.bridge.logic.social.retrieval.training import Query, RetrievalSystemBase, df


class TfIdfRetrieval(RetrievalSystemBase):
    def __init__(self, k=20):
        self.tfidf_vectorizer = sk.feature_extraction.text.TfidfVectorizer()
        self.tfidf_matrix = None
        self.k = k

    def train(self, df: pd.DataFrame):
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(df['text_preprocessed'])

    def retrieve(self, query: Query) -> list:
        query_vector = self.tfidf_vectorizer.transform([query.text])
        similarities = (query_vector * self.tfidf_matrix.T).toarray().flatten()
        similarities = similarities.argsort()[-self.k:][::-1]
        return [df['text'].iloc[i] for i in similarities]


print("training tfidf retrieval system")
tfidf_retrieval_system = TfIdfRetrieval()
tfidf_retrieval_system.train(df)
print("training tfidf retrieval system done")


def retrieve(query):
    return tfidf_retrieval_system.retrieve(Query(query))
