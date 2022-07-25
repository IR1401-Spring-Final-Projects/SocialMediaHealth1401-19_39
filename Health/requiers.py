# preprocess requirements
from __future__ import unicode_literals
from hazm import *
import nltk
import codecs
import json


normalizer = Normalizer()
stop_words = [normalizer.normalize(x.strip()) for x in codecs.open(
    'stopwords.txt', 'r', 'utf-8').readlines()]
stemmer = Stemmer()
lemmatizer = Lemmatizer()

# load the corpus (preprocessed)
all_tokens_nonstop_lemstem = None
with open('all_tokens_nonstop_lemstem.json', 'r') as input_file:
    all_tokens_nonstop_lemstem = json.load(input_file)
all_tokens_nonstop_lemstem = json.loads(all_tokens_nonstop_lemstem)

# load the corpus
bioset = None
f = open('bio.json')
bioset = json.load(f)
f.close()
