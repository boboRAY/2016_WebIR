import math
import xmltodict
import collections
import re
import numpy as np


RO_W = 0.8
TERM_W = 1
QUERY_K = 100
REL_K = 10


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
        if len(term) > 1:
            tf *= TERM_W
        d['docID'][line[0]] = tf
        v = doc_vector_list[doc_id]
        v[term] = tf * inverted_dict[term]['idf']
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
print('Inverted_Dict finished')


def gram(text):
    l = []
    for n in range(2, min(3, len(text))):
        for w in range(len(text)-(n-1)):
            l.append(text[w:w+n])
    v = {}
    for t in l:
        if t in v:
            v[t] += 1.0
        else:
            v[t] = 1.0
    return v


def unit_vector(qv):
    # normalize to unit vector
    square_len = 0
    for term, value in qv.items():
        square_len += value ** 2
    length = math.sqrt(square_len)
    for term, value in qv.items():
        qv[term] = value/length
    return qv


for idx, docv in enumerate(doc_vector_list()):
    doc_vector_list[idx] = unit_vector(docv)


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
    fd = {}
    for term, tf in d.items():
        if term in inverted_dict:
            fd[term] = tf * inverted_dict[term]['idf']
    return unit_vector(fd)


def get_top_k(qv, k):
    rank_dict = {}
    for term, value in qv.items():
        docs = inverted_dict[term]['docID']
        for docid, tfidf in docs.items():
            d = {}
            if docid in rank_dict:
                rank_dict[docid] += value * tfidf
            else:
                rank_dict[docid] = value * tfidf
    d = collections.Counter(rank_dict)
    d = d.most_common(k)
    return d


# get feedback
def get_feedback_vector(qv, k):
    global RO_W
    rel_d = get_top_k(qv, k)
    for d in rel_d:
        doc_id, s = d
        docv = doc_vector_list[int(doc_id)]
        for term, score in docv.items():
            score *= RO_W
            score /= k
            if term in qv:
                qv[term] += score
            else:
                qv[term] = score
    qv = unit_vector(qv)
    return qv


def make_ans():
    global RO_W, QUERY_K, REL_K
    ans_dict = {}
    for raw_query in test_list:
        query = raw_query['concepts']
        qv = get_vector(query)
        expanded_vector = get_feedback_vector(qv, REL_K)
        ans_dict[raw_query['number']] = get_top_k(expanded_vector, QUERY_K)
    ans_f = open('ans.txt', 'w')
    for number, ans_list in ans_dict.items():
        for ans in ans_list:
            docid, rank = ans
            doc = doc_list[int(docid)]
            docname = doc.lower().split('/')[3]
            answer = number[-3:] + ' ' + docname + '\n'
            ans_f.write(answer)
    ans_f.close()


def main():
    make_ans()

if __name__ == '__main__':
    main()


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
