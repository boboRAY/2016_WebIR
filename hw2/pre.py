# coding=UTF-8
from lib.porterStemmer import PorterStemmer
import os
import re
import json

# read stop word
f_stopwords = open('stop_words')
raw_stopwords = f_stopwords.read()
raw_stopwords = raw_stopwords.lower().splitlines()
stemmer = PorterStemmer()
stopwords = set()
for w in raw_stopwords:
    new_w = stemmer.stem(w, 0, len(w)-1)
    stopwords.add(w)


def get_tokens(path):
    # read raw data
    f = open(path, encoding='utf-8', errors='ignore')
    raw_str = f.read()

    # delete non-letters
    regex = re.compile('[^a-zA-Z0-9]')
    raw_str = regex.sub(' ', raw_str)

    # lowercase
    raw_str = raw_str.lower()

    # tokenize
    tokens = raw_str.split()
    # delete number
    tokens = [token for token in tokens if not token.isdigit()]

    # stemming
    stemmer = PorterStemmer()
    new_tokens = {}
    for token in tokens:
        new_token = stemmer.stem(token, 0, len(token)-1)
        if new_token not in stopwords and len(new_token) > 2:
            new_tokens[new_token] = new_tokens.get(new_token, 0) + 1
    return new_tokens

# list all file
train_docs = {}
t = 'Train/'
root = 'data/20news/' + t
for d in os.listdir(root):
    doc_ids = {}
    for fn in os.listdir(root+d):
        path = root + d + '/' + fn
        doc_ids[fn] = get_tokens(path)
    train_docs[d] = doc_ids

with open('pre/train.json', 'w') as fp:
    json.dump(train_docs, fp)

anss = open('ans.test').read().split()

test_docs = {}
t = 'Test/'
root = 'data/20news/' + t
for d in os.listdir(root):
    test_docs[d] = get_tokens(root + d)

with open('pre/test.json', 'w') as fp:
    json.dump(test_docs, fp)

unlabel_docs = {}
t = 'Unlabel/'
root = 'data/20news/' + t
for d in os.listdir(root):
    tokens = get_tokens(root+d)
    if len(tokens):
        unlabel_docs[d] = tokens
with open('pre/unlabel.json', 'w') as fp:
    json.dump(unlabel_docs, fp)

TRAIN_CLASE_DOCS_COUNTS = {}
for clase, docs in train_docs.items():
    TRAIN_CLASE_DOCS_COUNTS[clase] = len(docs)
TRAIN_DOCS_COUNT = sum(TRAIN_CLASE_DOCS_COUNTS.values())

# set up term ditc
term_clase_dict = {}  # term : {all_df, 'dfs': {clase: df}}
clase_all_tf = {clase: 0 for clase in TRAIN_CLASE_DOCS_COUNTS.keys()}
for clase, docs in train_docs.items():
    for d_id, tokens in docs.items():
        for token, tf in tokens.items():
            clase_all_tf[clase] += 1
            # term_clase_dict
            if token not in term_clase_dict:
                term_clase_dict[token] = {'all_tf': tf, 'tfs': {clase: tf}}
            else:
                # tfs
                term_clase_dict[token]['all_tf'] += tf
                if clase not in term_clase_dict[token]['tfs']:
                    term_clase_dict[token]['tfs'][clase] = tf
                else:
                    term_clase_dict[token]['tfs'][clase] += tf


with open('pre/train_term_clase.json', 'w') as fp:
    json.dump(term_clase_dict, fp)
with open('pre/train_all_tf.json', 'w') as fp:
    json.dump(clase_all_tf, fp)
