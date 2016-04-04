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

query = open('queries/query-test.xml', 'r').read()
q = xmltodict.parse(query)
query_list = q['xml']['topic']

# make inverted_dict: {term : 'docID' : { docID : tf}, 'idf' : idf}
term = ''
inverted_dict = {}
for lindex, line in enumerate(inverted_list):
    line = line.split()
    if(len(line) == 2):
        if term not in inverted_dict:
            continue
        d = inverted_dict[term]
        d['docID'][line[0]] = int(line[1])
        inverted_dict[term] = d
    else:
        df = int(line[2])
        idf = math.log10(total_file_count/df)
        term1 = vocab_list[int(line[0])]
        term2 = ''
        if term1 in stop_list or term2 in stop_list:
            continue
        if(int(line[1]) != -1):
            term2 = vocab_list[int(line[1])]
        term = term1 + term2
        if re.search('[a-zA-z]', term):
            continue
        inverted_dict[term] = {'idf': idf,
                               'docID': {}}


def get_query_vector(text):
    vector = {}
    for n in range(1, min(3, len(text))):
        for w in range(len(text)-(n-1)):
            word = text[w:w+n]
            if word in inverted_dict:
                idf = inverted_dict[word]['idf']
                if word in vector:
                    vector[word] = vector[word] + idf
                else:
                    vector[word] = idf

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

ans_dict = {}
for query in query_list:
    qt = ''
    for q, t in query.items():
        qt += t
    narrative = query['narrative']
    v = get_query_vector(qt)
    ans_dict[query['number']] = get_top_k(v, 100)

ans_f = open('ans.txt', 'w')
for number, ans_list in ans_dict.items():
    for ans in ans_list:
        docid, rank = ans
        doc = doc_list[int(docid)]
        docname = doc.lower().split('/')[3]
        answer = number[-3:] + ' ' + docname + '\n'
        ans_f.write(answer)
ans_f.close()
