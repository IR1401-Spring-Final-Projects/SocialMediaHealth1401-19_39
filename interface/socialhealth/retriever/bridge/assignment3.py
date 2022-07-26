from .logic.social.retrieval import boolean_retrieval as social_boolean

def boolean_search(subject: bool, search_term: str, expansion: bool):
    if subject: # health
        pass
    else: # social
        social_boolean.retrieve(search_term)


def tfidf_search(subject: bool, search_term: str, expansion: bool):
    if subject: # health
        pass
    else: # social
        pass


def transformer_search(subject: bool, search_term: str, expansion: bool):
    if subject: # health
        pass
    else: # social
        pass


def fasttext_search(subject: bool, search_term: str, expansion: bool):
    if subject: # health
        pass
    else: # social
        pass


def elasticsearch_search(subject: bool, search_term: str, expansion: bool):
    if subject: # health
        pass
    else: # social
        pass