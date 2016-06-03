# coding=UTF-8
import operator
import collections
import json
import math

term_clase_dict = json.load(open('pre/train_term_clase.json', 'r'))
clase_all_tf = json.load(open('pre/train_all_tf.json', 'r'))
test_tokens = json.load(open('pre/test.json'))
train_tokens = json.load(open('pre/train.json'))

TRAIN_CLASE_DOCS_COUNTS = {}
for clase, docs in train_tokens.items():
    TRAIN_CLASE_DOCS_COUNTS[clase] = len(docs)
TRAIN_DOCS_COUNT = sum(TRAIN_CLASE_DOCS_COUNTS.values())


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


def naive_bayes(tokens, parameters):
    probs = {clase: parameters[clase]['prior'] for clase in parameters.keys()}
    for token, tf in tokens.items():
        for clase in TRAIN_CLASE_DOCS_COUNTS.keys():
            probs[clase] += tf*parameters[clase]['terms'].get(token, 0)
    return max(probs.items(), key=operator.itemgetter(1))[0]

# test
answer_dict = {}
for tid, tokens in test_tokens.items():
    answer_dict[int(tid)] = naive_bayes(tokens, parameters)
f = open('nb_result', 'w')
for d in collections.OrderedDict(sorted(answer_dict.items())):
    f.write(str(d)+' '+answer_dict[d]+'\n')
f.close()

my_ans = open('nb_result', 'r').read().splitlines()
ans_test = open('ans.test').read().splitlines()

count = 0
for a, b in zip(my_ans, ans_test):
    if a == b:
        count += 1
print(count/len(ans_test))
