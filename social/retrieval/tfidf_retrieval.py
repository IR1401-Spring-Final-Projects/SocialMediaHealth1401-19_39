import sklearn as sk
import pandas as pd
from training import Query, RetrievalSystemBase
import training


class TfIdfRetrieval(RetrievalSystemBase):
    def __init__(self, k=10):
        self.tfidf_vectorizer = sk.feature_extraction.text.TfidfVectorizer()
        self.tfidf_matrix = None
        self.k = k

    def train(self, df: pd.DataFrame):
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(df['text_preprocessed'])

    def retrieve(self, query: Query) -> list:
        query_vector = self.tfidf_vectorizer.transform([query.text])
        similarities = (query_vector * self.tfidf_matrix.T).toarray().flatten()
        similarities = similarities.argsort()[-self.k:][::-1]
        return [training.df['text'].iloc[i] for i in similarities]


tfidf_retrieval_system = TfIdfRetrieval()
tfidf_retrieval_system.train(training.df)


def retrieve(query):
    return tfidf_retrieval_system.retrieve(query)
