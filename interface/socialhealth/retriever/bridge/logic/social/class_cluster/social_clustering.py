from sklearn.cluster import KMeans
import nltk
import functools
import pandas as pd
import string
from nltk.corpus import stopwords
import numpy as np
import itertools
import sklearn as sk
from sklearn.decomposition import TruncatedSVD
from scipy import sparse

stop_words = set(stopwords.words('english'))


# Load Data
PATH_TO_SENTIMENT140_DATASET = 'short-social-dataset.csv'
CSV_COLUMNS = ['target', 'id', 'date', 'flag', 'user', 'text']
TEST_SIZE = 0.2

df = pd.read_csv(PATH_TO_SENTIMENT140_DATASET, encoding='latin-1')
df.columns = CSV_COLUMNS
df.drop(columns=['target', 'id', 'date', 'flag', 'user'], inplace=True)

# Sample Data
df = df.sample(n=9999)

# Tokenize Data
df['text_tokenized'] = df['text'].apply(lambda x: nltk.word_tokenize(x))


# Normalize Data
def to_lower(tokens: list) -> list:
    """
    Converts the tokens to lower case.
    """
    return [token.lower() for token in tokens]


def contains_any_of(token: list, chars: str) -> bool:
    """
    Returns true if the token contains any of the characters in the given list.
    """
    return any(char in token for char in chars)


def remove_punctuation(tokens: list) -> list:
    """
    Removes punctuation from the given tokens.
    """
    return [token for token in tokens if not contains_any_of(token, string.punctuation + "’‘•")]


def remove_stop_words(tokens: list) -> list:
    """
    Removes stop words from the given tokens.
    """
    remove_stop_words.stop_words = set(nltk.corpus.stopwords.words('english'))
    return [token for token in tokens if token not in remove_stop_words.stop_words]


def normalize(tokens):
    """
    Normalizes the tokens of the lyrics.
    """
    normalization_functions = [to_lower, remove_punctuation, remove_stop_words]
    return functools.reduce(lambda x, f: f(x), normalization_functions, tokens)


df['text_normalized'] = df['text_tokenized'].apply(normalize)

# Stem Text
stemmer = nltk.stem.SnowballStemmer('english')
df['text_stemmed'] = df['text_normalized'].apply(lambda x: [stemmer.stem(t) for t in x])

# Join Text
df['text_preprocessed'] = df['text_stemmed'].apply(lambda x: ' '.join(x))

pd.set_option('display.max_columns', None)
# print(df.head())


tfidf_vectorizer = sk.feature_extraction.text.TfidfVectorizer(use_idf=True, norm ='l2', ngram_range=(1, 1))
tfidf_matrix = tfidf_vectorizer.fit_transform(df['text_preprocessed'])
vocabulary = tfidf_vectorizer.get_feature_names_out()


def process_query(query):
    splitted_input = query.split()
    nsi = []
    for x in splitted_input:
        if x not in stop_words:
            nsi.append([to_lower(x)])
    # print(nsi)
    return nsi


def find_query_vector(tokens):
    vector = np.zeros(len(vocabulary))
    for token in itertools.chain(*tokens):
        try:
            # print(str(token))
            temp = ""
            for c in token:
                temp += c
            try:
                index = tfidf_vectorizer.vocabulary_[temp]
                vector[index] = 1
            except:
                pass
        except ValueError:
            pass
    return vector


def cluster(query):
    query_vector = find_query_vector(process_query(query))
    query_vector = sparse.csr_matrix(query_vector)
    svd = TruncatedSVD(n_components=7, n_iter=7, random_state=42)

    x_dr = svd.fit_transform(tfidf_matrix)
    kmeans = KMeans(n_clusters=2, random_state=0).fit(tfidf_matrix)
    y = kmeans.fit_predict(x_dr)

    x_dr = svd.transform(query_vector)
    y = kmeans.predict(x_dr)

    return y[0]

