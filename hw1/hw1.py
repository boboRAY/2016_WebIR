import math
import xmltodict
import collections
import re


RO_W = 0.8
QUERY_K = 100
REL_K = 50
TOTAL_FILE_NUMBER = 46972

with open('model/inverted-file', 'r') as f:
    inverted_list = f.readlines()

with open('model/vocab.all') as f:
    vocab_list = f.read().splitlines()

with open('model/file-list', 'r') as f:
    doc_list = f.read().splitlines()

with open('stoplist', 'r') as f:
    stop_list = f.read().splitlines()

query = open('queries/query-test.xml', 'r').read()
q = xmltodict.parse(query)
test_list = q['xml']['topic']

doc_vector_list = [0]*len(doc_list)
for i in range(0, len(doc_list)):
    doc_vector_list[i] = {}

DOC_LEN_LIST = [0] * TOTAL_FILE_NUMBER
print('start make inverted')
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

        DOC_LEN_LIST[doc_id] += tf

        v = doc_vector_list[doc_id]
        v[term] = tf * inverted_dict[term]['idf']
        doc_vector_list[doc_id] = v

        d['docID'][doc_id] = tf * idf
        inverted_dict[term] = d
    else:
        df = float(line[2])
        idf = math.log10(TOTAL_FILE_NUMBER/df)
        term1 = vocab_list[int(line[0])]
        term2 = ''
        # bigram
        if(float(line[1]) != -1):
            term2 = vocab_list[int(line[1])]
            # if term1 in stop_list or term2 in stop_list:
            #     continue
        # unigram
        else:
            if term1 in stop_list:
                continue
            # if len(term1) < 2:
            #     continue
        term = term1 + term2
        term = term.lower()
        inverted_dict[term] = {'idf': idf,
                               'docID': {}}
print('Inverted_Dict finished')


def gram(text):
    l = []
    for n in range(1, min(3, len(text))):
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

AVG_DOC_LEN = sum(DOC_LEN_LIST)/float(len(DOC_LEN_LIST))
# okapi
for idx, docv in enumerate(doc_vector_list):
    for term, tf in docv.items():
        docv[term] = tf / (1 - 0.8 + 0.8 * DOC_LEN_LIST[idx]/AVG_DOC_LEN)
    doc_vector_list[idx] = docv


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
    fd_len = 0
    for term, tf in d.items():
        fd_len += tf

    for term, tf in d.items():
        if term in inverted_dict:
            # if len(term) > 1:
            #     tf *= TERM_W
            fd[term] = tf * inverted_dict[term]['idf']

    return fd


def get_top_k(qv, k):
    rank_dict = {}
    vector_lens = [0]*TOTAL_FILE_NUMBER
    for term, value in qv.items():
        docs = inverted_dict[term]['docID']
        for docid, tfidf in docs.items():
            vector_lens[docid] += tfidf ** 2
            if docid in rank_dict:
                rank_dict[docid] += value * tfidf
            else:
                rank_dict[docid] = value * tfidf
    for docid, rank in rank_dict.items():
        rank_dict[docid] /= math.sqrt(vector_lens[docid])
    d = {}
    d = collections.Counter(rank_dict)
    d = d.most_common(k)
    return d


# get feedback
def get_feedback_vector(qv, k):
    global RO_W
    rel_d = get_top_k(qv, k)
    for d in rel_d:
        doc_id, s = d
        docv = doc_vector_list[doc_id]
        for term, score in docv.items():
            score *= RO_W
            score /= k
            if term in qv:
                qv[term] += score
            else:
                qv[term] = score
    reduce_qv_list = collections.Counter(qv).most_common(500)
    reduce_qv = {}
    for tup in reduce_qv_list:
        term, tfidf = tup
        reduce_qv[term] = tfidf
    uv = unit_vector(reduce_qv)
    return uv


def make_ans():
    ans_dict = {}
    for raw_query in test_list:
        query = ''
        query += raw_query['concepts']
        query += ' '
        query += raw_query['concepts']
        query += ' '
        query += raw_query['title']
        query += ' '
        query += raw_query['narrative']
        query += ' '
        query += raw_query['question']
        qv = get_vector(query)
        ans_dict[raw_query['number']] = get_top_k(qv, QUERY_K)
        expanded_vector = get_feedback_vector(qv, REL_K)
        ans_dict[raw_query['number']] = get_top_k(expanded_vector, QUERY_K)
    ans_f = open('ans.txt', 'w')

    # write answer to file
    for number, ans_list in ans_dict.items():
        for ans in ans_list:
            docid, rank = ans
            doc = doc_list[docid]
            docname = doc.lower().split('/')[3]
            answer = number[-3:] + ' ' + docname + '\n'
            ans_f.write(answer)
    ans_f.close()


def main():
    make_ans()

if __name__ == '__main__':
    main()

