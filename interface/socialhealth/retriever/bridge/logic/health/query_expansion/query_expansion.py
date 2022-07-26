import codecs
import pickle
from hazm import *
from hazm import Normalizer

normalizer = Normalizer()
stopwords = [normalizer.normalize(x.strip()) for x in codecs.open(
    'stopwords.txt', 'r', 'utf-8').readlines()]


def query_expansion(query):
    queryList = query.split()
    expand = ''
    for x in queryList:
        if (x in stopwords):
            continue
        input_query = x
        pkl_file = open('data.pkl', 'rb')
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
                pass
                # list_of_synonyms[x] = x
        values = list(list_of_synonyms.values())
        try:
            expand += f'{values[0]} '
        except:
            expand += ''
    expand = expand.strip()
    print(expand)
    expandedQuery = f'{query} {expand}'
    return expandedQuery


if __name__ == '__main__':
    print(query_expansion('تپش قلب'))
