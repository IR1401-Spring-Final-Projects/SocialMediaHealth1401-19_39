import pandas as pd
import nltk
import string
import functools
from abc import ABC, abstractmethod

nltk.download('punkt')
nltk.download('stopwords')

# Load Data
PATH_TO_SENTIMENT140_DATASET = 'short-social-dataset.csv'
CSV_COLUMNS = ['target', 'id', 'date', 'flag', 'user', 'text']
TEST_SIZE = 0.2

df = pd.read_csv(PATH_TO_SENTIMENT140_DATASET, encoding='latin-1')
df.columns = CSV_COLUMNS
df.drop(columns=['target', 'id', 'date', 'flag', 'user'], inplace=True)

# Sample Data
df = df.sample(n=10000)

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


class Query:
    """
    A class that represents a query.
    """
    def __init__(self, text: str):
        self.text = text

    def __str__(self):
        return self.text

    def __repr__(self):
        return self.text


class RetrievalSystemBase(ABC):
    @abstractmethod
    def train(self, df: pd.DataFrame):
        pass

    @abstractmethod
    def retrieve(self, query: Query) -> list:
        pass
