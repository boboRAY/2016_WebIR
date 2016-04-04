import math
import xmltodict
import collections

with open('model/inverted-file', 'r') as f:
    inverted_list = f.readlines()

with open('model/vocab.all') as f:
    vocab_list = f.read().splitlines()

with open('model/file-list', 'r') as f:
    doc_list = f.read().splitlines()

total_file_count = 46972

query = open('queries/query-test.xml', 'r').read()
q = xmltodict.parse(query)
query_list = q['xml']['topic']
# OrderedDict([('number', 'CIRB010TopicZH001'), ('title', '集會遊行法與言論自由'), ('question', '查詢集會遊行法中有關主張共產主義或分裂國土規定之修正與討論。'), ('narrative', '相關文件內容應敘述集會遊行法原本對主張共產主義或分裂國土之限制，其是否符合憲法中對言論自由等基本人權的保障，大法官對此議題的相關解釋，學者專家的討論與看法，以及集會遊行法條文的修改現況。'), ('concepts', '集會遊行法、集會遊行、集遊法、憲法、言論自由、保障、共產主義、分裂國土、大法官會議、立法、修正條文。')])


# make inverted_dict: {term : { docID : tf}}
term = ''
inverted_dict = {}
for lindex, line in enumerate(inverted_list):
    line = line.split()
    if(len(line) == 2):
        d = inverted_dict[term]
        d['docID'][line[0]] = int(line[1])
        inverted_dict[term] = d
    else:
        index1 = int(line[0])
        index2 = int(line[1])
        df = int(line[2])
        idf = math.log10(total_file_count/df)
        if(index2 == -1):
            term = vocab_list[index1]
        else:
            term = vocab_list[index1] + vocab_list[index2]
        inverted_dict[term] = {'idf': idf,
                               'docID': {}}


def get_query_vector(text):
    vector = {}
    for n in range(1, min(3, len(text))):
        for w in range(len(text)-(n-1)):
            word = text[w:w+n]
            if word in vector:
                vector[word] = vector[word] + 1
            else:
                vector[word] = 1

    # normalize to unit vector
    square_len = 0
    for term, value in vector.items():
        square_len += value ** 2
    length = math.sqrt(square_len)
    for term, value in vector.items():
        vector[term] = value/length
    return vector


def get_top100(query):
    rank_dict = {}
    for term, value in query.items():
        if term not in inverted_dict:
            continue
        idf = inverted_dict[term]['idf']
        docs = inverted_dict[term]['docID']
        for docid, tf in docs.items():
            if docid in rank_dict:
                rank_dict[docid] = rank_dict[docid] + value*idf*tf
            else:
                rank_dict[docid] = value*idf*tf
        d = collections.Counter(rank_dict)
        d = d.most_common(100)
    return d


# OrderedDict([('number', 'CIRB010TopicZH001'), ('title', '集會遊行法與言論自由'), ('question', '查詢集會遊行法中有關主張共產主義或分裂國土規定之修正與討論。'), ('narrative', '相關文件內容應敘述集會遊行法原本對主張共產主義或分裂國土之限制，其是否符合憲法中對言論自由等基本人權的保障，大法官對此議題的相關解釋，學者專家的討論與看法，以及集會遊行法條文的修改現況。'), ('concepts', '集會遊行法、集會遊行、集遊法、憲法、言論自由、保障、共產主義、分裂國土、大法官會議、立法、修正條文。')])
ans_dict = {}
for query in query_list:
    qt = ''
    for q, t in query.items():
        qt += t
    narrative = query['narrative']
    v = get_query_vector(qt)
    ans_dict[query['number']] = get_top100(v)

ans_f = open('ans.txt', 'w')
for number, ans_list in ans_dict.items():
    for ans in ans_list:
        docid, rank = ans
        doc = doc_list[int(docid)]
        docname = doc.lower().split('/')[3]
        answer = number[-3:] + ' ' + docname + '\n'
        ans_f.write(answer)
ans_f.close()
