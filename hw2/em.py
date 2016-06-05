# coding=UTF-8
import operator
import collections
import json
import math


label_term_clase_dict = json.load(open('pre/train_term_clase.json', 'r'))
clase_all_tf = json.load(open('pre/train_all_tf.json', 'r'))

train_tokens = json.load(open('pre/train.json'))
test_tokens = json.load(open('pre/test.json'))
unlabel_tokens = json.load(open('pre/unlabel.json'))

LABEL_CLASE_DOCS_COUNTS = {}
for clase, docs in train_tokens.items():
    LABEL_CLASE_DOCS_COUNTS[clase] = len(docs)
LABEL_DOCS_COUNT = sum(LABEL_CLASE_DOCS_COUNTS.values())

u_c_d_prob = {clase: {} for clase in train_tokens.keys()}
l_c_d_prob = {path: {clase :{doc: float(int(path == clase)) for doc in train_tokens[path]} for clase in train_tokens} for path in train_tokens}

dictionary = set()
for t in label_term_clase_dict.keys():
    dictionary.add(t)
for d, ts in unlabel_tokens.items():
    for t in ts:
        dictionary.add(t)
TERMS_COUNT = len(dictionary)

clase_theta_dict = {clase: {"terms": {}, "prior": 0, "null": 0} for clase in train_tokens.keys()}

# use naive to get first u_c_d_prob
parameters = {clase: {'terms': {},
                      'prior':
                      math.log((LABEL_CLASE_DOCS_COUNTS[clase]+1)/(LABEL_DOCS_COUNT+20))}
              for clase in LABEL_CLASE_DOCS_COUNTS.keys()}

len_v = len(label_term_clase_dict)
for clase in clase_theta_dict:
    parameters[clase]['null'] = math.log(1/(clase_all_tf[clase]+len_v))
    for term, dic in label_term_clase_dict.items():
        tf = dic['tfs'].get(clase, 0)
        parameters[clase]['terms'][term] = math.log((1+tf)/(clase_all_tf[clase]+len_v))

# first e_step
for doc, tokens in unlabel_tokens.items():
    probs = {clase: 0 for clase in train_tokens.keys()}
    for clase in LABEL_CLASE_DOCS_COUNTS.keys():
        probs[clase] = parameters[clase]['prior']
        for token, tf in tokens.items():
            if token in label_term_clase_dict:
                probs[clase] += parameters[clase]['terms'][token] * tf
            elif token in dictionary:
                probs[clase] += parameters[clase]['null']
    k = max(probs.values())
    for clase, p in probs.items():
        if p-k < -30:
            probs[clase] = 0
        else:
            probs[clase] = math.exp(p-k)
    total = sum(probs.values())
    for clase, p in probs.items():
        p /= total
        u_c_d_prob[clase][doc] = p

# goal: update u_c_d_prob, l_c_d_prob
def e_step():
    global u_c_d_prob,l_c_d_prob, clase_theta_dict
    u_c_d_prob = {clase: {} for clase in train_tokens.keys()}
    l_c_d_prob = {path: {clase :{doc: float(int(path == clase)) for doc in train_tokens[path]} for clase in train_tokens} for path in train_tokens}
    for doc, tokens in unlabel_tokens.items():
        probs = {clase: math.log(clase_theta_dict[clase]['prior']) for clase in train_tokens.keys()}
        for token, tf in tokens.items():
            for clase in train_tokens.keys():
                if token in clase_theta_dict[clase]['terms']:
                    probs[clase] += math.log(clase_theta_dict[clase]['terms'][token]) * tf
                else:
                    probs[clase] += math.log(clase_theta_dict[clase]['null'])
        k = max(probs.values())
        for clase, p in probs.items():
            if p-k < -20:
                probs[clase] = 0
            else:
                probs[clase] = math.exp(p-k)
        total = sum(probs.values())
        for clase, p in probs.items():
            p /= total
            u_c_d_prob[clase][doc] = p

    for c, docs in train_tokens.items():
        for doc, tokens in docs.items():
            probs = {clase: math.log(clase_theta_dict[clase]['prior']) for clase in train_tokens.keys()}
            for token, tf in tokens.items():
                if token in clase_theta_dict[clase]['terms']:
                    probs[clase] += math.log(clase_theta_dict[clase]['terms'][token]) * tf
                else:
                    probs[clase] += math.log(clase_theta_dict[clase]['null'])
            k = max(probs.values())
            for clase, p in probs.items():
                if p-k < -20:
                    probs[clase] = 0
                else:
                    probs[clase] = math.exp(p-k)
            total = sum(probs.values())
            for clase, p in probs.items():
                p /= total
                l_c_d_prob[c][clase][doc] = p

# goal: update clase_theta_dict
def m_step():
    global clase_theta_dict, TERMS_COUNT, u_c_d_prob, l_c_d_prob
    clase_theta_dict = {clase: {"terms": {}, "prior": 0, "null": 0} for clase in train_tokens.keys()}
    # unlabel
    for clase, _ in u_c_d_prob.items():  # class
        down = TERMS_COUNT
        # prior = 1 + LABEL_CLASE_DOCS_COUNTS[clase]
        prior = 1
        # unlabel
        for doc, tokens in unlabel_tokens.items():
            prior += u_c_d_prob[clase][doc]
            for term, tf in tokens.items():
                up1 = tf*u_c_d_prob[clase][doc]
                if term in clase_theta_dict[clase]["terms"]:
                    clase_theta_dict[clase]["terms"][term] += up1
                else:
                    clase_theta_dict[clase]["terms"][term] = up1
                down += up1
        # label
        for c, docs in train_tokens.items():
            for doc, tokens in docs.items():
                prior += l_c_d_prob[c][clase][doc]
                for term, tf in tokens.items():
                    up2 = tf*l_c_d_prob[c][clase][doc]
                    if term in clase_theta_dict[clase]["terms"]:
                        clase_theta_dict[clase]["terms"][term] += up2
                    else:
                        clase_theta_dict[clase]["terms"][term] = up2
                        down += up2
        # for term, dic in label_term_clase_dict.items():
        #     up2 = dic['tfs'].get(clase, 0)
        #     if term in clase_theta_dict[clase]["terms"]:
        #         clase_theta_dict[clase]["terms"][term] += up2
        #     else:
        #         clase_theta_dict[clase]["terms"][term] = up2
        #     down += up2
        for term, up in clase_theta_dict[clase]["terms"].items():
            up += 1
            up /= down
            clase_theta_dict[clase]["terms"][term] = up
        clase_theta_dict[clase]["null"] = 1/down
        #prior
        clase_theta_dict[clase]["prior"] = prior/(20 + len(unlabel_tokens) + LABEL_DOCS_COUNT)


def naive_bayes(tokens):
    probs = {clase: math.log(clase_theta_dict[clase]['prior']) for clase in clase_theta_dict.keys()}
    for token, tf in tokens.items():
        for clase in LABEL_CLASE_DOCS_COUNTS.keys():
            if token in clase_theta_dict[clase]["terms"]:
                probs[clase] += tf*math.log(clase_theta_dict[clase]['terms'][token])
            elif token in dictionary:
                probs[clase] += tf*clase_theta_dict[clase]["null"]
    return max(probs.items(), key=operator.itemgetter(1))[0]


def test():
    # test
    answer_dict = {}
    for tid, tokens in test_tokens.items():
        answer_dict[int(tid)] = naive_bayes(tokens)
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

m_step()
for i in range(1, 50):
    print("e_step")
    e_step()
    print("m_step")
    m_step()
    test()
