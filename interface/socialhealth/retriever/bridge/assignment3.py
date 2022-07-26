from .logic.social.retrieval import boolean_retrieval as social_boolean
from .logic.social.retrieval import fasttext_retrieval as social_fasttext
from .logic.social.retrieval import tfidf_retrieval as social_tfidf
from .logic.social.retrieval import transformer_retrieval as social_transformer
from .logic.social.query_exansion import query_expansion as social_expansion
from .logic.social.retrieval import elastic_retrieval as social_elastic



def boolean_search(subject: bool, search_term: str, expansion: bool):
    if subject: # health
        pass
    else: # social
        if expansion:
            expanded_query = f'{search_term} {social_expansion.query_expansion(search_term)[0]}'
            social_boolean.retrieve(expanded_query)
        else:
            social_boolean.retrieve(search_term)


def tfidf_search(subject: bool, search_term: str, expansion: bool):
    if subject: # health
        pass
    else: # social
        if expansion:
            expanded_query = f'{search_term} {social_expansion.query_expansion(search_term)[0]}'
            social_tfidf.retrieve(expanded_query)
        else:
            social_tfidf.retrieve(search_term)


def transformer_search(subject: bool, search_term: str, expansion: bool):
    if subject: # health
        pass
    else: # social
        if expansion:
            expanded_query = f'{search_term} {social_expansion.query_expansion(search_term)[0]}'
            social_transformer.retrieve(expanded_query)
        else:
            social_transformer.retrieve(search_term)


def fasttext_search(subject: bool, search_term: str, expansion: bool):
    if subject: # health
        pass
    else: # social
        if expansion:
            expanded_query = f'{search_term} {social_expansion.query_expansion(search_term)[0]}'
            social_fasttext.retrieve(expanded_query)
        else:
            social_fasttext.retrieve(search_term)


def elasticsearch_search(subject: bool, search_term: str, expansion: bool):
    if subject: # health
        pass
    else: # social
        if expansion:
            expanded_query = f'{search_term} {social_expansion.query_expansion(search_term)[0]}'
            social_elastic.retrieve(expanded_query)
        else:
            social_elastic.retrieve(search_term)
            