from elasticsearch import Elasticsearch
from elasticsearch import helpers
from abc import ABC, abstractmethod
import pandas as pd
import json
from .requires import *


class Query:
    """
    A class that represents a query.
    """

    def __init__(self, text: str):
        self.text = text

    def __str__(self):
        return self.text

    def __repr__(self):
        return self.text


class RetrievalSystemBase(ABC):
    @abstractmethod
    def train(self, df: pd.DataFrame):
        pass

    @abstractmethod
    def retrieve(self, query: Query) -> list:
        pass


class ElasticsearchRetrieval(RetrievalSystemBase):
    def __init__(self):
        config = json.load(open('config.json'))
        address = config['elastic_address']
        self.es = Elasticsearch(address)
        self.index = config['elastic_index']
        if not self.es.ping():
            print(self.es.info())
            raise Exception('Could not connect to Elasticsearch')

    def train(self, df: pd.DataFrame):
        documents = []
        for index, row in df.iterrows():
            documents.append(
                {'_index': self.index, '_title': row['title'], '_source': row.to_dict()})
        helpers.bulk(self.es, documents)
        self.es.indices.refresh(index=self.index)

    def retrieve(self, query: Query) -> list:
        results = self.es.search(index=self.index, query={
                                 'match': {'text': query.text}})
        return [result['_source'] for result in results['hits']['hits']]


def main(search_term):
    # Example usage
    CSV_COLUMNS = ['tags', 'categories', 'title',
                   'abstract', 'paragraphs', 'link']
    df = pd.read_json('bio.json')
    df.to_csv()
    df.columns = CSV_COLUMNS
    esr = ElasticsearchRetrieval()
    esr.train(df)
    answer = esr.retrieve(Query(search_term))
    print(answer)
    return answer


def retrieve(search_term):
    return main(search_term)
