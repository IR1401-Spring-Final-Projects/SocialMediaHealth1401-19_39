from .logic.social.retrieval import boolean_retrieval as social_boolean
from .logic.social.retrieval import fasttext_retrieval as social_fasttext
from .logic.social.retrieval import tfidf_retrieval as social_tfidf
from .logic.social.retrieval import transformer_retrieval as social_transformer
from .logic.social.query_exansion import query_expansion as social_expansion
from .logic.social.retrieval import elastic_retrieval as social_elastic

from .logic.health.retrieval import boolean_retrieval as health_boolean
from .logic.health.retrieval import tfidf_retrieval as health_tfidf
from .logic.health.retrieval import transformer_retrieval as health_transformer
from .logic.health.retrieval import fasttext_retrieval as health_fasttext
from .logic.health.retrieval import elastic_retrieval as health_elastic
from .logic.health.query_expansion import query_expansion as health_expansion


def boolean_search(subject: bool, search_term: str, expansion: bool):
    if subject:  # health
        if expansion:
            expanded_query = f'{search_term} {health_expansion.query_expansion(search_term)[0]}'
            health_boolean.retrieve(expanded_query)
        else:
            health_boolean.retrieve(search_term)
    else:  # social
        if expansion:
            expanded_query = f'{search_term} {social_expansion.query_expansion(search_term)[0]}'
            social_boolean.retrieve(expanded_query)
        else:
            social_boolean.retrieve(search_term)


def tfidf_search(subject: bool, search_term: str, expansion: bool):
    if subject:  # health
        if expansion:
            expanded_query = f'{search_term} {health_expansion.query_expansion(search_term)[0]}'
            health_tfidf.retrieve(expanded_query)
        else:
            health_tfidf.retrieve(search_term)
    else:  # social
        if expansion:
            expanded_query = f'{search_term} {social_expansion.query_expansion(search_term)[0]}'
            social_tfidf.retrieve(expanded_query)
        else:
            social_tfidf.retrieve(search_term)


def transformer_search(subject: bool, search_term: str, expansion: bool):
    if subject:  # health
        if expansion:
            expanded_query = f'{search_term} {health_expansion.query_expansion(search_term)[0]}'
            health_transformer.retrieve(expanded_query)
        else:
            health_transformer.retrieve(search_term)
    else:  # social
        if expansion:
            expanded_query = f'{search_term} {social_expansion.query_expansion(search_term)[0]}'
            social_transformer.retrieve(expanded_query)
        else:
            social_transformer.retrieve(search_term)


def fasttext_search(subject: bool, search_term: str, expansion: bool):
    if subject:  # health
        if expansion:
            expanded_query = f'{search_term} {health_expansion.query_expansion(search_term)[0]}'
            health_fasttext.retrieve(expanded_query)
        else:
            health_fasttext.retrieve(search_term)
    else:  # social
        if expansion:
            expanded_query = f'{search_term} {social_expansion.query_expansion(search_term)[0]}'
            social_fasttext.retrieve(expanded_query)
        else:
            social_fasttext.retrieve(search_term)


def elasticsearch_search(subject: bool, search_term: str, expansion: bool):
    if subject:  # health
        if expansion:
            expanded_query = f'{search_term} {health_expansion.query_expansion(search_term)[0]}'
            health_elastic.retrieve(expanded_query)
        else:
            health_elastic.retrieve(search_term)
    else:  # social
        if expansion:
            expanded_query = f'{search_term} {social_expansion.query_expansion(search_term)[0]}'
            social_elastic.retrieve(expanded_query)
        else:
            social_elastic.retrieve(search_term)
