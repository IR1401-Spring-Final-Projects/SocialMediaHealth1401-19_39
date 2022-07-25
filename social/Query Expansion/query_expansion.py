import random
from nltk.corpus import wordnet as wn


def query_expansion(query: str) -> list[str]:
    query = query
    query_list = [i.lower() for i in query.split()]

    query_syn = {}
    ret = {}

    for word in query_list:
        query_syn[word] = []
        for syn in wn.synsets(word):
            query_syn[word].append(syn.lemmas()[0].name().lower())
        ret[word] = [*set(query_syn[word])]
        try:
            ret[word].remove(word)
        except:
            pass
    print(ret)
    expanded_queries = []
    for i in range(3):
        temp = ""
        for word in query_list:
            if len(ret[word]) > 0:
                temp += random.choice(ret[word]) + " "
            else:
                temp += word + " "

        expanded_queries.append(temp[:-1])

    return expanded_queries


print(query_expansion("hungry victims prefer death to chain"))
