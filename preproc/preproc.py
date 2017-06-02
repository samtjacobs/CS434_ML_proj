from gensim.models import Word2Vec
import pandas as pd
from string import punctuation
import sys
import numpy as np
# Load pretrained model (since intermediate data is not included, the model cannot be refined with additional data)
model = Word2Vec.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)


def get_words(sentence):
    vec_seq = []
    try:
        for word in ''.join(c for c in sentence if c not in punctuation and not c.isdigit()).split():
#    for word in sentence:
            if word in model:
                vec_seq.append(model[word])
    except TypeError:
        print("saw float in: " + str(sentence))
    return vec_seq


data = pd.read_csv(sys.argv[1])
output = []
trs = 'train' in sys.argv[1]
for sentences in zip(data['question1'], data['question2']):
    nr = [get_words(sentences[0]), get_words(sentences[1])]
    output.append(nr)

np.save(sys.argv[1].split('.')[0] + '.npy', np.asarray(output))
if trs:
    np.save('labels.npy', np.asarray(data['is_duplicate']))
