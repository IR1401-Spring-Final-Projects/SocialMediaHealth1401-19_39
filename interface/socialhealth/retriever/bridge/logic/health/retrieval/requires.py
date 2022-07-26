# preprocess requirements
from __future__ import unicode_literals
from hazm import *
import nltk
import codecs
from transformers import AutoConfig, AutoTokenizer, AutoModel, TFAutoModel, BigBirdModel
import json
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import pandas as pd

documents_length = 875

normalizer = Normalizer()
stop_words = [normalizer.normalize(x.strip()) for x in codecs.open(
    'stopwords.txt', 'r', 'utf-8').readlines()]
stopwords = [normalizer.normalize(x.strip()) for x in codecs.open(
    'stopwords.txt', 'r', 'utf-8').readlines()]
stemmer = Stemmer()
lemmatizer = Lemmatizer()

# load the corpus (preprocessed)
all_tokens_nonstop_lemstem = []
with open('all_tokens_nonstop_lemstem.json', 'r') as input_file:
    all_tokens_nonstop_lemstem = json.load(input_file)
all_tokens_nonstop_lemstem = json.loads(all_tokens_nonstop_lemstem)

with open('all_tokens_nonstop.json', 'r') as input_file:
    all_tokens_nonstop = json.load(input_file)
all_tokens_nonstop = json.loads(all_tokens_nonstop)


# load the corpus
bioset = None
f = open('bio.json')
bioset = json.load(f)
f.close()
# print(bioset[0])


vectorizer = TfidfVectorizer(
    use_idf=True, norm='l2', ngram_range=(1, 2), analyzer='word')
doc_term_mat = vectorizer.fit_transform(
    [' '.join(doc) for doc in all_tokens_nonstop_lemstem])
vocabulary = vectorizer.get_feature_names_out()

# MODEL_NAME = "SajjadAyoubi/distil-bigbird-fa-zwnj"
# transformer_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


with open('transformer.pickle', 'rb') as handle:
    averaged_vectors = pickle.load(handle)

# with open('transformer_model.pickle', 'rb') as handle:
#     transformer_model = pickle.load(handle)


"""
CSV_COLUMNS = ['tags', 'categories', 'title', 'abstract', 'paragraphs', 'link']
df = pd.read_json('bio.json')
df.to_csv()
df.columns = CSV_COLUMNS
esr = ElasticsearchRetrieval()
esr.train(df)
print(esr.retrieve(Query('نپش قلب')))
"""


# print(df[0:10])
