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
    regex = re.compile('[^a-zA-Z]')
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
    for n in range(len(tokens)):
        token = tokens[n]
        new_token = stemmer.stem(token, 0, len(token)-1)
        tokens[n] = new_token

    new_tokens = [x for x in tokens if len(x) > 2]

    # stemming for stop word
    # remove stop word from tokens
    for n, stopword in enumerate(stopwords):
        new_stopword = stemmer.stem(stopword, 0, len(stopword)-1)
        while new_stopword in new_tokens:
            new_tokens.remove(new_stopword)
    return new_tokens

# set up term ditc
term_clase_dict = {}  # term : {all_df, 'dfs': {clase: df}}
for clase, docs in train_docs.items():
    clase_dict = {'tf_sum': 0}
    for d_id, p in docs.items():
        tokens = get_tokens(p)
        term_in_p = set()
        for t in tokens:
            # term_clase_dict
            if t not in term_clase_dict:
                term_clase_dict[t] = {'all_df': 1, 'all_tf': 1,
                                      'tfs': {clase: 1}, 'dfs': {clase: 1}}
            else:
                # tfs
                term_clase_dict[t]['all_tf'] += 1
                if clase not in term_clase_dict[t]['tfs']:
                    term_clase_dict[t]['tfs'][clase] = 1
                else:
                    term_clase_dict[t]['tfs'][clase] += 1
                # dfs
                if t not in term_in_p:
                    term_clase_dict[t]['all_df'] += 1
                if clase not in term_clase_dict[t]['dfs']:
                    term_clase_dict[t]['dfs'][clase] = 1
                else:
                    if t not in term_in_p:
                        term_clase_dict[t]['dfs'][clase] += 1
            term_in_p.add(t)

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

# feature selection
FEATURE_COUNT = 1000
feature_set = set()
rounds = list(train_docs.keys())
turn = 0
# for clase in train_docs.keys():
#     terms = collections.Counter(llr_dict[clase])
#     feature_set = feature_set.union(set(dict(terms.most_common(30)).keys()))

while len(feature_set) < FEATURE_COUNT:
    clase = rounds[turn]
    terms = llr_dict[clase]
    new_feature = max(terms.items(), key=operator.itemgetter(1))[0]
    if new_feature not in feature_set:
        turn = (turn + 1) % 20
    feature_set.add(new_feature)
    del llr_dict[clase][new_feature]
    # turn = (turn + 1) % 20

clase_feature_tf_dict = {clase: 0 for clase in train_docs.keys()}
for t in feature_set:
    for clase, num in term_clase_dict[t]['tfs'].items():
        clase_feature_tf_dict[clase] += num


def df_naive_bayes(doc_path, feature_set):
    # get doc vector
    vector = set()
    tokens = get_tokens(doc_path)
    for t in tokens:
        if t in feature_set:
            vector.add(t)

    probs = {clase: 0.0 for clase in train_docs.keys()}
    # term_clase_dict = {}  term : {all_df, 'dfs': {clase: df}}
    for t in feature_set:
        # for clase, df in term_clase_dict[t]['dfs'].items():
        for clase in train_docs.keys():
            prob = 0
            if clase in term_clase_dict[t]['dfs']:
                df = term_clase_dict[t]['dfs'][clase] + 1
            else:
                df = 1
            prob = df/(TRAIN_CLASE_DOCS_COUNTS[clase]+2)
            if t in vector:
                prob = math.log(prob)
            else:
                prob = math.log(1-prob)
            probs[clase] += prob
    for clase, num in TRAIN_CLASE_DOCS_COUNTS.items():
        probs[clase] += math.log(num/TRAIN_DOCS_COUNT)
    return max(probs.items(), key=operator.itemgetter(1))[0]

# list all file
test_docs = {}
root = 'data/20news/Test/'
answer_dict = {}
for d in os.listdir(root):
    answer_dict[int(d)] = df_naive_bayes(root+d, feature_set)
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
