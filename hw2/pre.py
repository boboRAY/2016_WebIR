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
    new_tokens = []
    for token in tokens:
        new_token = stemmer.stem(token, 0, len(token)-1)
        if new_token not in stopwords and len(new_token) > 3:
            new_tokens.append(new_token)

    return new_tokens

# set up term ditc
term_clase_dict = {}  # term : {all_df, 'dfs': {clase: df}}
clase_all_tf = {clase: 0 for clase in TRAIN_CLASE_DOCS_COUNTS.keys()}
for clase, docs in train_docs.items():
    for d_id, p in docs.items():
        tokens = get_tokens(p)
        term_in_p = set()
        for t in tokens:
            clase_all_tf[clase] += 1
            # term_clase_dict
            if t not in term_clase_dict:
                term_clase_dict[t] = {'all_tf': 1, 'tfs': {clase: 1}}
            else:
                # tfs
                term_clase_dict[t]['all_tf'] += 1
                if clase not in term_clase_dict[t]['tfs']:
                    term_clase_dict[t]['tfs'][clase] = 1
                else:
                    term_clase_dict[t]['tfs'][clase] += 1
                term_in_p.add(t)


def train_parameter():
    len_v = len(term_clase_dict)
    clase_theta_dict = {clase: {'terms': {},
                        'prior':
                       math.log((TRAIN_CLASE_DOCS_COUNTS[clase]+1)/(TRAIN_DOCS_COUNT+20))}
                        for clase in TRAIN_CLASE_DOCS_COUNTS.keys()}
    for term, dic in term_clase_dict.items():
        for clase in clase_theta_dict:
            tf = dic['tfs'].get(clase, 0)
            clase_theta_dict[clase]['terms'][term] = math.log((1+tf)/(clase_all_tf[clase] + len_v))
    return clase_theta_dict


parameters = train_parameter()


def naive_bayes(doc_path, parameters):
    tokens = get_tokens(doc_path)
    probs = {clase: parameters[clase]['prior'] for clase in parameters.keys()}
    for token in tokens:
        for clase in TRAIN_CLASE_DOCS_COUNTS.keys():
            probs[clase] += parameters[clase]['terms'].get(token, 0)
    return max(probs.items(), key=operator.itemgetter(1))[0]

# list all file
test_docs = {}
root = 'data/20news/Test/'
answer_dict = {}
for d in os.listdir(root):
    answer_dict[int(d)] = naive_bayes(root+d, parameters)
f = open('nb_result', 'w')
for d in collections.OrderedDict(sorted(answer_dict.items())):
    f.write(str(d)+' '+answer_dict[d]+'\n')
f.close()

my_ans = open('nb_result', 'r').read().splitlines()
ans_test = open('data/ans.test').read().splitlines()

count = 0
for a, b in zip(my_ans, ans_test):
    if a == b:
        count += 1
print(count/len(ans_test))
