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


inverted_dict = {}
AVG_DOC_LEN = 0
DOC_LEN_LIST = [0]*len(doc_list)
doc_vector_lens = [None]*len(doc_vector_list)
# make inverted_dict: {term : 'docID' : { docID : tf}, 'idf' : idf}
term = ''
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
            if re.search('[a-zA-Z]', term1) or re.search('[a-zA-Z]', term2):
                continue
        else:
            if term1.isdigit():
                continue
        if term1.isdigit() and term2.isdigit():
            continue
        if term1 in stop_list and term2 in stop_list:
            continue
        term = term1 + term2
        term = term.lower()
        inverted_dict[term] = {'idf': idf,
                               'docID': {}}

# get doc length and vector length
for idx, dv in enumerate(doc_vector_list):
    dlen = 0
    for term, tf in dv.items():
        DOC_LEN_LIST[idx] += tf
        dlen += tf ** 2
    doc_vector_lens[idx] = math.sqrt(dlen)
for dlen in DOC_LEN_LIST:
    AVG_DOC_LEN += dlen
AVG_DOC_LEN /= total_file_count


# okapi
def okapi(b, tf, doc_id):
    doc_len = DOC_LEN_LIST[doc_id]
    tf = tf / (1 - b + b * doc_len / AVG_DOC_LEN)
    return tf


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
    s = re.sub('[a-zA-Z]', '', s)
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


def get_top_k(oka_w , qv, k):
    rank_dict = {}
    for term, value in qv.items():
        idf = inverted_dict[term]['idf']
        docs = inverted_dict[term]['docID']
        for docid, tf in docs.items():
            tf = okapi(oka_w, tf, int(docid))
            tfidf = idf*tf
            d = {}
            if docid in rank_dict:
                rank_dict[docid] = rank_dict[docid] + value*tfidf
            else:
                rank_dict[docid] = value*tfidf
    for doc_id, score in rank_dict.items():
        rank_dict[doc_id] = score/doc_vector_lens[int(doc_id)]
    d = collections.Counter(rank_dict)
    d = d.most_common(k)
    return d


# get feedback
def get_feedback_vector(oka_w, v, weight, k):
    uv = unit_vector(v)
    rel_d = get_top_k(oka_w, uv, k)
    for d in rel_d:
        doc_id, s = d
        docv = unit_vector(doc_vector_list[int(doc_id)])
        for term, score in docv.items():
            score *= weight
            score /= k
            if term in v:
                uv[term] += score
            else:
                uv[term] = score
    uv = unit_vector(uv)
    return v


def make_ans(oka_w, ro_w, term_w, rel_k, k):
    ans_dict = {}
    for raw_query in test_list:
        query = raw_query['concepts']
        queryv = get_vector(query)
        expanded_vector = get_feedback_vector(oka_w, queryv, ro_w, rel_k)
        for term, score in expanded_vector.items():
            if re.search('[a-zA-Z]', term):
                score = score * term_w
            elif len(term) >= 2:
                score = score * term_w
            expanded_vector[term] = score
        v = unit_vector(expanded_vector)
        ans_dict[raw_query['number']] = get_top_k(oka_w, v, k)
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
    make_ans(0.5, 1, 1, 10, 100)


# if __name__ == '__main__':
main()

# for training

with open('queries/ans-train') as f:
    real_ans = f.read().splitlines()


def get_map(oka_w, ro_w, term_w, rel_k, k):
    # use new vector to get top 100 list
    ans_dict = {}
    for raw_query in train_list:
        query = raw_query['concepts']
        queryv = get_vector(query)
        expanded_vector = get_feedback_vector(oka_w, queryv, ro_w, rel_k)
        for term, score in expanded_vector.items():
            if re.search('[a-zA-Z]', term):
                score = score * term_w
            elif len(term) >= 2:
                score = score * term_w
            expanded_vector[term] = score
        v = unit_vector(expanded_vector)
        ans_dict[raw_query['number']] = get_top_k(oka_w, v, k)
    # ans_f = open('ans.txt', 'w')
    scores = []
    for number, anses in ans_dict.items():
        count = 0
        right = 0
        score = 0
        for ans in anses:
            count += 1
            docid, rank = ans
            doc = doc_list[int(docid)]
            docname = doc.lower().split('/')[3]
            answer = number[-3:] + ' ' + docname
            if answer in real_ans:
                right += 1
                score += right/count
        score /= right
        scores.append(score)
    average = 0
    for s in scores:
        average += s
    average /= k
    return average


# # # train
# para = {'match': 0}
# for ro_w in range(1, 10, 1):
#     for term_w in np.arange(1.0, 2.1, 0.2):
#         for okapi_b in np.arange(0.1, 1, 0.1):
#             s = get_map(okapi_b, ro_w, term_w, 20, 10)
#             print(ro_w, term_w, okapi_b, s)
#             if s > para['match']:
#                 para['match'] = s
#                 para['ro_w'] = ro_w
#                 para['term_w'] = term_w
#                 para['okapi_b'] = okapi_b
# print(para)
