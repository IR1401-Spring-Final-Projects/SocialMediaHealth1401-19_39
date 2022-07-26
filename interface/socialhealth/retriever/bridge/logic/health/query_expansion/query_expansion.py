import pickle
from hazm import *


def query_expansion(query):
    queryList = query.split()
    expand = ''
    for x in queryList:
        input_query = x
        pkl_file = open('SocialMediaHealth1401-19_39/interface/socialhealth/retriever/bridge/logic/health'
                        '/query_expansion /data.pkl', 'rb')
        dic = pickle.load(pkl_file)
        pkl_file.close()
        normalizer = Normalizer()
        normalized = normalizer.normalize(input_query)
        stemmer = Stemmer()
        stemmed = stemmer.stem(normalized)
        tokens = word_tokenize(stemmed)
        list_of_synonyms = {}
        for x in tokens:
            try:
                tokens_synonyms = word_tokenize(dic[x])
                for y in tokens_synonyms:
                    list_of_synonyms[x] = y
            except:
                list_of_synonyms[x] = x
        values = list(list_of_synonyms.values())
        expand += f'{values[0]} '
    expand = expand.strip()
    print(expand)
    expandedQuery = f'{query} {expand}'
    return expandedQuery


if __name__ == '__main__':
    print(query_expansion('تپش قلب'))
