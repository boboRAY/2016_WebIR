# coding=UTF-8
from lib.porterStemmer import PorterStemmer
import os
import re
import math
import operator
import collections


# list all file
train_docs = {}
# for t in ['Train/', 'Test/', 'Unlabel/']:
t = 'Train/'
root = 'data/20news/' + t
for d in os.listdir(root):
    doc_ids = {}
    for fn in os.listdir(root+d):
        path = root + d + '/' + fn
        doc_ids[fn] = path
    train_docs[d] = doc_ids

TRAIN_CLASE_DOCS_COUNTS = {}
for clase, docs in train_docs.items():
    TRAIN_CLASE_DOCS_COUNTS[clase] = len(docs)
TRAIN_DOCS_COUNT = sum(TRAIN_CLASE_DOCS_COUNTS.values())


def get_tokens(path):
    # read raw data
    f = open(path, encoding='utf-8', errors='ignore')
    raw_str = f.read()

    # delete non-letters
    regex = re.compile('[^a-zA-Z0-9]')
    raw_str = regex.sub(' ', raw_str)

    # read stop word
    f_stopwords = open('stop_words')
    raw_stopwords = f_stopwords.read()
    stopwords = raw_stopwords.lower().splitlines()

    # lowercase
    raw_str = raw_str.lower()

    # tokenize
    tokens = raw_str.split()

    # stemming
    stemmer = PorterStemmer()
    for n, token in enumerate(tokens):
        new_token = stemmer.stem(token, 0, len(token)-1)
        tokens[n] = new_token

    # stemming for stop word
    # remove stop word from tokens
    for n, stopword in enumerate(stopwords):
        new_stopword = stemmer.stem(stopword, 0, len(stopword)-1)
        while new_stopword in tokens:
            tokens.remove(new_stopword)
    return tokens

# set up term ditc
clase_term_dict = {}  # clase : {doc_num, tf_sum, terms : {df, all_tf ,term:[]}}
term_clase_dict = {}  # term : {all_df, 'dfs': {clase: df}}
for clase, docs in train_docs.items():
    clase_dict = {'tf_sum': 0}
    for d_id, p in docs.items():
        tokens = get_tokens(p)
        term_in_p = set()
        for t in tokens:
            # term_clase_dict
            if t not in term_clase_dict:
                term_clase_dict[t] = {'all_df': 1, 'dfs': {clase: 1}}
            else:
                if t not in term_in_p:
                    term_clase_dict[t]['all_df'] += 1
                if clase not in term_clase_dict[t]['dfs']:
                    term_clase_dict[t]['dfs'][clase] = 1
                else:
                    if t not in term_in_p:
                        term_clase_dict[t]['dfs'][clase] += 1

            # clase_term_dict
            clase_dict['tf_sum'] += 1  # total term count in clase +1
            if t not in clase_dict:
                clase_dict[t] = {'df': 1, 'tf': 1, 'docs': {p: 1}}
            else:
                clase_dict[t]['tf'] += 1
                docs = clase_dict[t]['docs']
                if t not in term_in_p:
                    clase_dict[t]['df'] += 1
                    docs[p] = 1
                else:
                    docs[p] += 1
                    clase_dict[t]['docs'] = docs
            term_in_p.add(t)
    clase_term_dict[clase] = clase_dict

CLASE_COUNT = len(TRAIN_CLASE_DOCS_COUNTS)
llr_dict = {clase: {} for clase in train_docs.keys()}
# LLR : save in clase_term_dict[clase][term][llr]
for term, t_dict in term_clase_dict.items():
    for clase, df in t_dict['dfs'].items():
        N = TRAIN_DOCS_COUNT
        all_df = t_dict['all_df']
        n11 = float(df)
        n01 = float(all_df-n11)
        n10 = float(TRAIN_CLASE_DOCS_COUNTS[clase]-n11)
        n00 = float(N-TRAIN_CLASE_DOCS_COUNTS[clase]-n01)
        h11 = math.log(((n11+n01)/N)**n11)
        h12 = math.log((1-(n11+n01)/N)**n10)
        h13 = math.log(((n11+n01)/N)**n01)
        h14 = math.log((1-(n11+n01)/N)**n00)
        h21 = math.log((n11/(n11+n10))**n11)
        h22 = math.log((1-n11/(n11+n10))**n10)
        h23 = math.log((n01/(n01+n00))**n01)
        h24 = math.log((1-(n01/(n01+n00)))**n00)
        h1 = h11+h12+h13+h14
        h2 = h21+h22+h23+h24
        l_ratio = -2*(h1-h2)
        llr_dict[clase][term] = l_ratio

# backup_llr_dict = llr_dict
# feature selection
# FEATURE_COUNT = len(term_clase_dict.keys())//40
FEATURE_COUNT = 500
feature_set = set()
rounds = list(train_docs.keys())
turn = 0
while len(feature_set) < FEATURE_COUNT:
    clase = rounds[turn]
    terms = llr_dict[clase]
    new_feature = max(terms.items(), key=operator.itemgetter(1))[0]
    if new_feature not in feature_set:
        turn = (turn + 1) % 20
    feature_set.add(new_feature)
    del llr_dict[clase][new_feature]
    # turn = (turn + 1) % 20


def naive_bayes(doc_path):
    # get doc vector
    vector = set()
    tokens = get_tokens(doc_path)
    for t in tokens:
        if t in feature_set:
            vector.add(t)

    probs = {clase: 0.0 for clase in train_docs.keys()}
    # term_clase_dict = {}  term : {all_df, 'dfs': {clase: df}}
    for t in vector:
        for clase, df in term_clase_dict[t]['dfs'].items():
            probs[clase] += math.log(df/TRAIN_CLASE_DOCS_COUNTS[clase])
    for clase, num in TRAIN_CLASE_DOCS_COUNTS.items():
        probs[clase] += math.log(num/TRAIN_DOCS_COUNT)
    return max(probs.items(), key=operator.itemgetter(1))[0]


# list all file
test_docs = {}
root = 'data/20news/Test/'
answer_dict = {}
for d in os.listdir(root):
    answer_dict[int(d)] = naive_bayes(root+d)
f = open('nb_result', 'w')
for d in collections.OrderedDict(sorted(answer_dict.items())):
    f.write(str(d)+' '+answer_dict[d]+'\n')
f.close()

my_ans = open('nb_result','r').read().splitlines()
ans_test = open('data/ans.test').read().splitlines()

count = 0
for a, b in zip(my_ans,ans_test):
    if a == b:
        count += 1
print(count)
