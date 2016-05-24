# coding=UTF-8
from lib.porterStemmer import PorterStemmer
import os
import re

# list all file
p = []
for t in ['Train', 'Test', 'Unlabel']:
    root = 'data/20news/' + t
    dirs = [x[0] for x in os.walk(root)]
    f = open(t+'_list', 'w')
    for d in dirs:
        for fn in os.listdir(d):
            path = d + '/' + fn
            if os.path.isfile(path):
                p.append(path)
                f.write(path+'\n')
    f.close()

with open('Train_list', 'r') as f:
    train_list = f.read().split()


def get_tokens(path):
    # read raw data
    f = open(path)
    raw_str = f.read()

    # delete non-letters
    regex = re.compile('[^a-zA-Z0-9]')
    raw_str = regex.sub(' ', raw_str)

    # read stop word
    f_stopwords = open('stop_words')
    raw_stopwords = f_stopwords.read()
    stopwords = raw_stopwords.lower().splitlines()

    # lowercase
    raw_str = raw_str.lower()

    # tokenize
    tokens = raw_str.split()

    # stemming
    stemmer = PorterStemmer()
    for n, token in enumerate(tokens):
        new_token = stemmer.stem(token, 0, len(token)-1)
        tokens[n] = new_token

    # stemming for stop word
    # remove stop word from tokens
    for n, stopword in enumerate(stopwords):
        new_stopword = stemmer.stem(stopword, 0, len(stopword)-1)
        while new_stopword in tokens:
            tokens.remove(new_stopword)
    return tokens


# set up dictionary
words_set = set()
for p in train_list:
    print(p)
    words_set.union(set(get_tokens(p)))
