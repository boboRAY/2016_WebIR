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

doc_dict = {doc : 0 for doc in unlabel_tokens.keys()}
u_c_d_prob = {clase: doc_dict for clase in train_tokens.keys()}


dictionary = set()
for t in label_term_clase_dict.keys():
    dictionary.add(t)
for d, ts in unlabel_tokens.items():
    for t in ts:
        dictionary.add(t)
TERMS_COUNT = len(dictionary)

clase_theta_dict = {clase : {"terms": {}, "prior": 0} for clase in train_tokens.keys()}

# use naive to get first u_c_d_prod
len_v = len(label_term_clase_dict)
parameters = {clase: {'terms': {},
                            'prior':
                            (LABEL_CLASE_DOCS_COUNTS[clase]+1)/(LABEL_DOCS_COUNT+20)}
                    for clase in LABEL_CLASE_DOCS_COUNTS.keys()}

for term, dic in label_term_clase_dict.items():
    for clase in parameters:
        tf = dic['tfs'].get(clase, 0)
        parameters[clase]['terms'][term] = (1+tf)/(clase_all_tf[clase] + len_v)

for doc, tokens in unlabel_tokens.items():
    for token, tf in tokens.items():
        for clase in LABEL_CLASE_DOCS_COUNTS.keys():
            u_c_d_prob[clase][doc] += tf*parameters[clase]['terms'].get(token, 0)



# goal: update u_c_d_prob
def e_step():
    global u_c_d_prob, clase_theta_dict
    for clase, dic in clase_theta_dict.items():
        t_c_prob = dic['terms']
        prior = dic['prior']
        total = 0
        for doc, tokens in unlabel_tokens.items():
            up = prior
            for term, tf in tokens.items():
                up *= t_c_prob[term] ** tf
            u_c_d_prob[clase][doc] = up
            total += up
        for doc, pd in u_c_d_prob[clase].items():
            u_c_d_prob[clase][doc] /= total


# goal: update clase_theta_dict
def m_step():
    global clase_theta_dict, TERMS_COUNT
    doc_count = len(unlabel_tokens) + LABEL_DOCS_COUNT + 20
    # unlabel
    for clase, terms in u_c_d_prob.items():  # class
        down = TERMS_COUNT
        prior = 1 + LABEL_CLASE_DOCS_COUNTS[clase]
        # unlabel
        for doc, terms in unlabel_tokens.items():
            prior += u_c_d_prob[clase][doc]
            for term, tf in terms.items():
                p = tf*u_c_d_prob[clase][doc] + clase_theta_dict[clase]["terms"].get(term, 0)
                # if p != 0:
                clase_theta_dict[clase]["terms"][term] = p
                down += p
        for term, dic in label_term_clase_dict.items():
            p = clase_theta_dict[clase]['terms'].get(term, 0) + dic['tfs'].get(clase, 0)
            # if p != 0:
            clase_theta_dict[clase]["terms"][term] = p
            down += p
        for term, tp in clase_theta_dict[clase]["terms"].items():
            tp += 1
            tp /= down
            clase_theta_dict[clase]["terms"][term] = tp

        #prior
        clase_theta_dict[clase]["prior"] = (prior + 1)/(TERMS_COUNT + len(unlabel_tokens) + LABEL_DOCS_COUNT)



def naive_bayes(tokens, parameters):
    probs = {clase: parameters[clase]['prior'] for clase in parameters.keys()}
    for token, tf in tokens.items():
        for clase in LABEL_CLASE_DOCS_COUNTS.keys():
            probs[clase] += tf*parameters[clase]['terms'].get(token, 0)
    return max(probs.items(), key=operator.itemgetter(1))[0]

def test():
    # test
    answer_dict = {}
    for tid, tokens in test_tokens.items():
        answer_dict[int(tid)] = naive_bayes(tokens, clase_theta_dict)
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
for i in range(1, 10):
    e_step()
    m_step()
    test()
