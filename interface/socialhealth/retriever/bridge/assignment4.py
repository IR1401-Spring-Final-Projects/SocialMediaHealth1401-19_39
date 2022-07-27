from .logic.health.retrieval import classification as health_classify
from .logic.health.retrieval import clustering as health_cluster


def classify_cluster(subject: bool, search_term: str):
    if subject:  # health
        return health_classify.run(search_term), health_cluster.run(search_term)
    else:  # social
        pass
