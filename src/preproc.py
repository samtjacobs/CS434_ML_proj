from gensim.models import Word2Vec
import pandas as pd
from string import punctuation
import cPickle as pickle
import sys
# Load pretrained model (since intermediate data is not included, the model cannot be refined with additional data)
model = Word2Vec.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)


def get_words(sentence):
    vec_seq = []
    for word in ''.join(c for c in sentence not in punctuation):
        if word in model:
            vec_seq.append(model[word])
    return vec_seq


data = pd.read_csv(sys.argv[1])
output = []
trs = 'train' in sys.argv[1]
if trs:
    dts = zip(data['question1'],data['question2'],data['is_duplicate'])
else:
    dts = zip(data['question1'], data['question2'])

for sentences in dts:
    nr = [get_words(sentences[0]), get_words(sentences[1])]
    if trs:
        nr.append(sentences[2])
    output.append(nr)

pickle.dump(output, open(sys.argv[1].split('.')[0] + '.p', "wb"))
