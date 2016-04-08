import math
import xmltodict
import collections
import re
import numpy as np

with open('model/inverted-file', 'r') as f:
    inverted_list = f.readlines()

with open('model/vocab.all') as f:
    vocab_list = f.read().splitlines()

with open('model/file-list', 'r') as f:
    doc_list = f.read().splitlines()

with open('stoplist', 'r') as f:
    stop_list = f.read().splitlines()

total_file_count = 46972

query = open('queries/query-train.xml', 'r').read()
q = xmltodict.parse(query)
train_list = q['xml']['topic']

query = open('queries/query-test.xml', 'r').read()
q = xmltodict.parse(query)
test_list = q['xml']['topic']


doc_vector_list = [0]*len(doc_list)
for i in range(0, len(doc_list)):
    doc_vector_list[i] = {}

del_voc = []
del_voc = del_voc + list(range(1900, 1935)) + list(range(1936, 1995)) +\
          list(range(12334,  12363)) + list(range(12365, 12451)) +\
          list(range(12452, 12454))

# make inverted_dict: {term : 'docID' : { docID : tf}, 'idf' : idf}
term = ''
inverted_dict = {}
for lindex, line in enumerate(inverted_list):
    line = line.split()
    if(len(line) == 2):
        # some terms were abandoned
        if term not in inverted_dict:
            continue
        d = inverted_dict[term]
        doc_id = int(line[0])
        tf = float(line[1])
        d['docID'][line[0]] = float(line[1])
        v = doc_vector_list[doc_id]
        if len(term) > 1:
            tf *= 2
        v[term] = tf
        inverted_dict[term] = d
    else:
        if int(line[0]) in del_voc or int(line[1]) in del_voc:
            continue
        df = float(line[2])
        idf = math.log10(total_file_count/df)
        term1 = vocab_list[int(line[0])]
        term2 = ''
        if(float(line[1]) != -1):
            term2 = vocab_list[int(line[1])]
            if term1.isdigit() and term2.isdigit():
                continue
            if re.search('[a-zA-Z]', term1) or re.search('[a-zA-Z]', term2):
                continue
        else:
            if term1.isdigit():
                continue
        if term1 in stop_list and term2 in stop_list:
            continue
        term = term1 + term2
        term = term.lower()
        inverted_dict[term] = {'idf': idf,
                               'docID': {}}


def get_doc_len(doc_id):
    dv = doc_vector_list[doc_id]
    dlen = 0
    for term, tf in dv.items():
        dv[term] = tf
        dlen += tf ** 2
    return math.sqrt(dlen)


def gram(text):
    l = []
    for n in range(2, min(3, len(text))):
        for w in range(len(text)-(n-1)):
            l.append(text[w:w+n])
    v = {}
    for t in l:
        if t in v:
            v[t] += 1
        else:
            v[t] = 1
    return v


def get_vector(s):
    s += ' '
    text = ''
    d = {}
    was_en = False
    # seperate English and Chinese word
    for c in s:
        if was_en:
            if re.search('[a-zA-z]', c):
                text += c
            else:
                text = text.lower()
                if text in d:
                    d[text] += 1
                else:
                    d[text] = 1
                text = ''
                was_en = False
        else:
            if re.search('[a-zA-Z]', c):
                text += c
                was_en = True
    # gram
    s = re.sub('[a-zA-Z0-9]', '', s)
    d = {**gram(s), **d}
    return d


def unit_vector(qv):
    vector = {}
    # make feature vector with tfidf
    for word, tf in qv.items():
        if word in inverted_dict:
            idf = inverted_dict[word]['idf']
            vector[word] = tf*idf

    # normalize to unit vector
    square_len = 0
    for term, value in vector.items():
        square_len += value ** 2
    length = math.sqrt(square_len)
    for term, value in vector.items():
        vector[term] = value/length
    return vector


def get_top_k(qv, k):
    rank_dict = {}
    for term, value in qv.items():
        idf = inverted_dict[term]['idf']
        docs = inverted_dict[term]['docID']
        for docid, tf in docs.items():
            tfidf = idf*tf
            d = {}
            if docid in rank_dict:
                rank_dict[docid] = rank_dict[docid] + value*tfidf
            else:
                rank_dict[docid] = value*tfidf
    for doc_id, score in rank_dict.items():
        rank_dict[doc_id] = score/get_doc_len[int(doc_id)]
    d = collections.Counter(rank_dict)
    d = d.most_common(k)
    return d


# get feedback
def get_feedback_vector(query, weight, k):
    qv = get_vector(query)
    uqv = unit_vector(qv)
    rel_d = get_top_k(uqv, k)
    for d in rel_d:
        doc_id, s = d
        docv = doc_vector_list[int(doc_id)]
        udocv = unit_vector(docv)
        for term, score in udocv.items():
            score *= weight
            score /= k
            if term in v:
                v[term] += score
            else:
                v[term] = score
    return v


def make_ans(ro_w, term_w, rel_k, k):
    ans_dict = {}
    for raw_query in test_list:
        query = raw_query['concepts']
        expanded_vector = get_feedback_vector(query, ro_w, rel_k)
        # for term, score in expanded_vector.items():
        #     if re.search('[a-zA-Z]', term):
        #         score = score * term_w
        #     elif len(term) == 2:
        #         score = score * term_w
        #     expanded_vector[term] = score
        eqv = unit_vector(expanded_vector)
        ans_dict[raw_query['number']] = get_top_k(eqv, k)
    ans_f = open('ans.txt', 'w')
    for number, ans_list in ans_dict.items():
        for ans in ans_list:
            docid, rank = ans
            doc = doc_list[int(docid)]
            docname = doc.lower().split('/')[3]
            answer = number[-3:] + ' ' + docname + '\n'
            ans_f.write(answer)
    ans_f.close()

make_ans(5, 2, 10, 100)


# for training

# with open('queries/ans-train') as f:
#     real_ans = f.read().splitlines()


# def get_map(ro_w, term_w, rel_k, k):
#     # use new vector to get top 100 list
#     ans_dict = {}
#     for raw_query in train_list:
#         query = raw_query['concepts']
#         expanded_vector = get_feedback_vector(query, ro_w, rel_k)
#         for term, score in expanded_vector.items():
#             if re.search('[a-zA-Z]', term):
#                 score = score * term_w
#             elif len(term) == 2:
#                 score = score * term_w
#             expanded_vector[term] = score
#         v = unit_vector(expanded_vector)
#         ans_dict[raw_query['number']] = get_top_k(v, k)
#     # ans_f = open('ans.txt', 'w')
#     ans_list = []
#     scores = []
#     for number, anses in ans_dict.items():
#         count = 0
#         right = 0
#         score = 0
#         for ans in anses:
#             count += 1
#             docid, rank = ans
#             doc = doc_list[int(docid)]
#             docname = doc.lower().split('/')[3]
#             # answer = number[-3:] + ' ' + docname + '\n'
#             answer = number[-3:] + ' ' + docname
#             if answer in real_ans:
#                 right += 1
#                 score += right/count
#             ans_list.append(answer)
#         score /= right
#         scores.append(score)
#         # ans_f.write(answer)
#     # ans_f.close()
#     average = 0
#     for s in scores:
#         average += s
#     average /= k
#     return average


# # train
# para = {'match': 0}
# for ro_w in range(5, 10, 1):
#     for term_w in np.arange(1.0, 2.1, 0.2):
#         for rel_k in range(10, 20, 1):
#             s = get_map(ro_w, term_w, 10, 10)
#             print(ro_w, term_w, s)
#             if s > para['match']:
#                 para['match'] = s
#                 para['ro_w'] = ro_w
#                 para['term_w'] = term_w
#                 para['rel_k'] = rel_k
# print(para)
