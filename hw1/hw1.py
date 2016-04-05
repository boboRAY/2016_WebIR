import math
import xmltodict
import collections
import re

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


# make inverted_dict: {term : 'docID' : { docID : tf}, 'idf' : idf}
term = ''
inverted_dict = {}
for lindex, line in enumerate(inverted_list):
    line = line.split()
    if(len(line) == 2):
        if term not in inverted_dict:
            continue
        d = inverted_dict[term]
        d['docID'][line[0]] = float(line[1])
        inverted_dict[term] = d
    else:
        weight = 1.0
        df = float(line[2])
        idf = math.log10(total_file_count/df)
        term1 = vocab_list[int(line[0])]
        term2 = ''
        if term1 in stop_list or term2 in stop_list:
            continue
        if(float(line[1]) != -1):
            term2 = vocab_list[int(line[1])]
        term = term1 + term2
        if re.search('[0-9]', term):
            continue
        inverted_dict[term] = {'idf': idf,
                               'docID': {}}


def gram(text):
    l = []
    for n in range(1, min(3, len(text))):
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
    flag = False
    # seperate en and chinese word
    for c in s:
        if flag:
            if re.search('[a-zA-z]', c):
                text += c
            else:
                if text in d:
                    d[text] += 1
                else:
                    d[text] = 1
                text = ''
                flag = False
        else:
            if re.search('[a-zA-Z]', c):
                text += c
                flag = True
    # gram
    s = re.sub('[a-zA-Z]', '', s)
    d = {**gram(s), **d}
    return d


def unit_vector(qv):
    vector = {}
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


def get_top_k(query, k):
    rank_dict = {}
    doc_len_dict = {}
    for term, value in query.items():
        if term not in inverted_dict:
            continue
        idf = inverted_dict[term]['idf']
        docs = inverted_dict[term]['docID']
        for docid, tf in docs.items():
            tfidf = idf*tf
            d = {}
            if docid in rank_dict:
                doc_len_dict[docid] = doc_len_dict[docid] + tfidf**2
                rank_dict[docid] = rank_dict[docid] + value*tfidf
            else:
                doc_len_dict[docid] = tfidf ** 2
                rank_dict[docid] = value*tfidf
    for docid, score in rank_dict.items():
        rank_dict[docid] = score/math.sqrt(doc_len_dict[docid])
    d = collections.Counter(rank_dict)
    d = d.most_common(k)
    return d


# get feedback
def get_feedback_vector(query, weight):
    qt = query['concepts']
    v = get_vector(qt)
    uv = unit_vector(v)
    rel_d = get_top_k(uv, 10)
    for d in rel_d:
        doc, s = d
        doc_f = open(doc_list[int(doc)], 'r').read()
        q = xmltodict.parse(doc_f)
        trains = q['xml']['doc']['text']['p']
        doc_s = ''
        try:
            for t in trains:
                doc_s += t
        except:
            pass
        fv = get_vector(doc_s)
        for term, score in fv.items():
            if re.search('[0-9]', term):
                continue
            score *= weight
            if term in v:
                v[term] += score
            else:
                v[term] = score
    return v

with open('queries/ans-train') as f:
    real_ans = f.read().splitlines()


def make_ans(ro_w, term_w, k):
    ans_dict = {}
    for query in test_list:
        expanded_vector = get_feedback_vector(query, ro_w)
        for term, score in expanded_vector.items():
            if re.search('[a-zA-Z]', term):
                score = score * term_w
            elif len(term) == 2:
                score = score * term_w
            expanded_vector[term] = score
        v = unit_vector(expanded_vector)
        ans_dict[query['number']] = get_top_k(v, k)
    ans_f = open('ans.txt', 'w')
    for number, ans_list in ans_dict.items():
        for ans in ans_list:
            docid, rank = ans
            doc = doc_list[int(docid)]
            docname = doc.lower().split('/')[3]
            answer = number[-3:] + ' ' + docname + '\n'
            ans_f.write(answer)
    ans_f.close()

make_ans(9, 2, 100)


# for training
def get_map(ro_w, term_w, k):
    # use new vector to get top 100 list
    ans_dict = {}
    for query in train_list:
        expanded_vector = get_feedback_vector(query, ro_w)
        for term, score in expanded_vector.items():
            if re.search('[a-zA-Z]', term):
                score = score * term_w
            elif len(term) == 2:
                score = score * term_w
            expanded_vector[term] = score
        v = unit_vector(expanded_vector)
        ans_dict[query['number']] = get_top_k(v, k)
    # ans_f = open('ans.txt', 'w')
    ans_list = []
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
            # answer = number[-3:] + ' ' + docname + '\n'
            answer = number[-3:] + ' ' + docname
            if answer in real_ans:
                right += 1
                score += right/count
            ans_list.append(answer)
        score /= k
        scores.append(score)
        # ans_f.write(answer)
    # ans_f.close()
    average = 0
    for s in scores:
        average += s
    average /= 10
    return average


# train
# para = {'match': 0}
# for ro_w in range(1, 10 ,1):
#     for term_w in np.arange(1.0, 2.1, 0.2):
#         s = get_map(ro_w, term_w, 20)
#         print(ro_w, term_w, s)
#         if s > para['match']:
#             para['match'] = s
#             para['ro_w'] = ro_w
#             para['term_w'] = term_w
# print(para)
