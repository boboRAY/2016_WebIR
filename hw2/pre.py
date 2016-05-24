from lib.porterStemmer import PorterStemmer
import os

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

TRAIN_F = open('Train_list','r')
TEST_F = open('Test_list','r')
UNLABEL_F = open('Unlabel_list','r')

# read raw data 
f = open('28.txt')
raw_str = f.read()

# remove special character
raw_str = raw_str.replace("'",'').replace('.','').replace(',','')

# read stop word
f_stopwords = open('stop_words.txt')
raw_stopwords = f_stopwords.read()
stopwords = raw_stopwords.lower().splitlines()


# lowercase
raw_str = raw_str.lower()

#tokenize
tokens = raw_str.split()

#stemming
stemmer = PorterStemmer()
for n,token in enumerate(tokens):
    new_token = stemmer.stem(token,0,len(token)-1)
    tokens[n] = new_token    
    
# stemming for stop word
# remove stop word from tokens
for n,stopword in enumerate(stopwords):
    new_stopword = stemmer.stem(stopword,0,len(stopword)-1)
    while new_stopword in tokens:
        tokens.remove(new_stopword)
'''
#save result as txt file
result = open('result.txt','w')
for  token in tokens:
    result.write(token+'\n')
result.close()
'''
