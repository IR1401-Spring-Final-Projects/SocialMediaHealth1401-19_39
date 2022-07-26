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
        config = json.load(open('config.json'))
        address = config['elastic_address']
        self.es = Elasticsearch(address)
        self.index = config['elastic_index']
        if not self.es.ping():
            print(self.es.info())
            raise Exception('Could not connect to Elasticsearch')

    def train(self, df: pd.DataFrame):
        documents = []
        for _, row in df.iterrows():
            documents.append({'_index': self.index, '_source': row.to_dict()})
        helpers.bulk(self.es, documents)
        self.es.indices.refresh(index=self.index)

    def retrieve(self, query: Query) -> list:
        results = self.es.search(index=self.index, query={'multi_match': {'query': query.text, 'fields': []}}, size=10000)
        return [result['_source'] for result in results['hits']['hits']]


def main():
    # Example usage
    CSV_COLUMNS = ['target', 'id', 'date', 'flag', 'user', 'text']
    df = pd.read_csv("../interface/socialhealth/short-social-dataset.csv")
    print(len(df.index))
    df.columns = CSV_COLUMNS
    esr = ElasticsearchRetrieval()
    esr.train(df)
    print((esr.retrieve(Query('birthday'))))


if __name__ == '__main__':
    main()
