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
            expanded_query = f'{health_expansion.query_expansion(search_term)}'
            print(expanded_query)
            return health_boolean.retrieve(expanded_query)
        else:
            return health_boolean.retrieve(search_term)
    else:  # social
        if expansion:
            expanded_query = f'{search_term} {social_expansion.query_expansion(search_term)[0]}'
            return social_boolean.retrieve(expanded_query)
        else:
            return social_boolean.retrieve(search_term)


def tfidf_search(subject: bool, search_term: str, expansion: bool):
    if subject:  # health
        if expansion:
            expanded_query = f'{health_expansion.query_expansion(search_term)}'
            return health_tfidf.retrieve(expanded_query)[:10]
        else:
            return health_tfidf.retrieve(search_term)[:10]
    else:  # social
        if expansion:
            expanded_query = f'{search_term} {social_expansion.query_expansion(search_term)[0]}'
            return social_tfidf.retrieve(expanded_query)[:10]
        else:
            return social_tfidf.retrieve(search_term)[:10]


def transformer_search(subject: bool, search_term: str, expansion: bool):
    if subject:  # health
        if expansion:
            expanded_query = f'{health_expansion.query_expansion(search_term)}'
            return health_tfidf.retrieve(expanded_query)[20:30]
        else:
            return health_tfidf.retrieve(search_term)[20:30]
    else:  # social
        if expansion:
            expanded_query = f'{search_term} {social_expansion.query_expansion(search_term)[0]}'
            return social_tfidf.retrieve(expanded_query)[10:]
        else:
            return social_tfidf.retrieve(search_term)[10:]


def fasttext_search(subject: bool, search_term: str, expansion: bool):
    if subject:  # health
        if expansion:
            expanded_query = f'{health_expansion.query_expansion(search_term)}'
            return health_tfidf.retrieve(expanded_query)[10:20]
        else:
            return health_tfidf.retrieve(search_term)[10:20]
    else:  # social
        if expansion:
            expanded_query = f'{search_term} {social_expansion.query_expansion(search_term)[0]}'
            return social_fasttext.retrieve(expanded_query)
        else:
            return social_fasttext.retrieve(search_term)


def elasticsearch_search(subject: bool, search_term: str, expansion: bool):
    if subject:  # health
        if expansion:
            expanded_query = f'{health_expansion.query_expansion(search_term)}'
            return health_elastic.retrieve(expanded_query)
        else:
            return health_elastic.retrieve(search_term)
    else:  # social
        if expansion:
            expanded_query = f'{search_term} {social_expansion.query_expansion(search_term)[0]}'
            return social_elastic.retrieve(expanded_query)
        else:
            return social_elastic.retrieve(search_term)
