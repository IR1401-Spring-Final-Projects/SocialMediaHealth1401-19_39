from elasticsearch import Elasticsearch
from elasticsearch import helpers
from abc import ABC, abstractmethod
import pandas as pd
import json


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
        address = "http://192.168.100.43:9200"
        self.es = Elasticsearch(address)
        self.index = "health_final"
        if not self.es.ping():
            print(self.es.info())
            raise Exception('Could not connect to Elasticsearch')

    def train(self, df: pd.DataFrame):
        pass

    def retrieve(self, query: Query) -> list:
        results = self.es.search(index=self.index, query={'multi_match': {'query': query.text, 'fields': []}}, size=10)
        return [result['_source'] for result in results['hits']['hits']]

try:
    esr = ElasticsearchRetrieval()
except:
    print("Could not connect to Elasticsearch")

def retrieve(search_term):
    results = esr.retrieve(Query(search_term))
    return [f"{r['title']}:{r['link']}" for r in results]
