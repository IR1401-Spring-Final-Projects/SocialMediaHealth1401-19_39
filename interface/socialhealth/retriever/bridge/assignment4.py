from .logic.health.retrieval import classification as health_classify
from .logic.health.retrieval import clustering as health_cluster
from .logic.social.class_cluster import social_classification as social_classify
from .logic.social.class_cluster import social_clustering as social_cluster


def classify_cluster(subject: bool, search_term: str):
    if subject:  # health
        return health_classify.run(search_term), health_cluster.run(search_term)
    else:  # social
        return social_classify.classification(search_term), social_cluster.cluster(search_term)
