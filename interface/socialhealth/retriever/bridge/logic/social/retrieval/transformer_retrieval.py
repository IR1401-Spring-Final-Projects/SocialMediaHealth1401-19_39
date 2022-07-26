import pandas as pd
from simpletransformers.retrieval import RetrievalModel
from retriever.bridge.logic.social.retrieval.training import Query, RetrievalSystemBase, df


class TransformerRetrieval(RetrievalSystemBase):
    def __init__(self, k=10):
        self.k = 10

    def train(self, df: pd.DataFrame):
        model_type = "dpr"
        context_encoder_name = "facebook/dpr-ctx_encoder-single-nq-base"
        question_encoder_name = "facebook/dpr-question_encoder-single-nq-base"

        args = {
            "include_title": False,
        }
        self.model = RetrievalModel(
            model_type=model_type,
            context_encoder_name=context_encoder_name,
            query_encoder_name=question_encoder_name,
            args=args,
            use_cuda=False,
        )
        self.train_df = df[['text', 'text_preprocessed']].copy(deep=True)
        self.train_df.rename(columns={'text': 'query_text', 'text_preprocessed': 'gold_passage'}, inplace=True)
        self.model.train_model(self.train_df)

    def retrieve(self, query: Query) -> list:
        to_predict = [query.text]
        prediction_passages = self.train_df.copy(deep=True)
        prediction_passages['title'] = ['']*len(prediction_passages)
        prediction_passages.rename(columns={'query_text': 'passages'}, inplace=True)
        predicted_passages, _, _, _ = self.model.predict(to_predict, prediction_passages=prediction_passages, retrieve_n_docs=self.k)
        return predicted_passages[0]


print("training transformer retrieval system")
transformer_retrieval = TransformerRetrieval()
#transformer_retrieval.train(df)
print("training transformer retrieval system done")


def retrieve(query):
    return transformer_retrieval.retrieve(Query(query))
