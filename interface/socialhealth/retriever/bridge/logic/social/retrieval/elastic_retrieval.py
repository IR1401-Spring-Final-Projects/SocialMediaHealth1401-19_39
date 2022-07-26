import pandas as pd
from retriever.bridge.logic.social.retrieval.training import Query, RetrievalSystemBase, df
from elasticsearch import Elasticsearch
from elasticsearch import helpers
import pandas as pd
import json

class ElasticsearchRetrieval(RetrievalSystemBase):
    def __init__(self):
        address = "http://192.168.100.43:9200"
        self.es = Elasticsearch(address)
        self.index = "social_final_final"
        if not self.es.ping():
            print(self.es.info())
            raise Exception('Could not connect to Elasticsearch')

    def train(self, df: pd.DataFrame):
        pass

    def retrieve(self, query: Query) -> list:
        results = self.es.search(index=self.index, query={'multi_match': {'query': query.text, 'fields': []}}, size=10000)
        return [result['_source'] for result in results['hits']['hits']]

try:
    elastic_retrieval = ElasticsearchRetrieval()
except Exception as e:
    print(e)


def retrieve(query):
    results = elastic_retrieval.retrieve(Query(query))
    return [r['text'] for r in results]
    