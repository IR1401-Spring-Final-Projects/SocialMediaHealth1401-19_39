import pandas as pd
from retriever.bridge.logic.social.retrieval.training import Query, RetrievalSystemBase, df
from query import Query
from elasticsearch import Elasticsearch
from elasticsearch import helpers
import pandas as pd
import json

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
            documents.append({'_index': self.index, '_id': row['id'], '_source': row.to_dict()})
        helpers.bulk(self.es, documents)
        self.es.indices.refresh(index=self.index)

    def retrieve(self, query: Query) -> list:
        results = self.es.search(index=self.index, query={'match': {'text': query.text}})
        return [result['_source'] for result in results['hits']['hits']]

elastic_retrieval = ElasticsearchRetrieval()
elastic_retrieval.train(df)

def retrieve(query):
    return ElasticsearchRetrieval.retrieve(Query(query))
    