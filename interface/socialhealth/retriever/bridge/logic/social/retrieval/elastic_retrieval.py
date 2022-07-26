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
        self.index = "social"
        if not self.es.ping():
            print(self.es.info())
            raise Exception('Could not connect to Elasticsearch')

    def train(self, df: pd.DataFrame):
        documents = []
        for index, row in df.iterrows():
            documents.append({'_index': self.index, '_id': row['id'], '_source': row.to_dict()})
        helpers.bulk(self.es, documents)
        self.es.indices.refresh(index=self.index)

    def retrieve(self, query: Query) -> list:
        results = self.es.search(index=self.index, query={'match': {'text': query.text}})
        return [result['_source'] for result in results['hits']['hits']]

try:
    elastic_retrieval = ElasticsearchRetrieval()
    # elastic_retrieval.train(df)
except Exception as e:
    print(e)

def retrieve(query):
    return ElasticsearchRetrieval.retrieve(Query(query))
    