from gensim.models import KeyedVectors
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from string import punctuation
import sys
import numpy as np
# Load pretrained model (since intermediate data is not included, the model cannot be refined with additional data)
word2vec_mod = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

def build_tfidf_dic(data):
    # Presumes data is list of words/sentences
    corpus = []
    for item in data:
        for word in item.split():
            corpus.append(word.translate(None, punctuation).lower())
    tfidf = TfidfVectorizer()
    tfidf.fit_transform(corpus)
    return dict(zip(tfidf.get_feature_names(), tfidf.idf_))


def get_words(sentence, tfidf_dic):
    # Modified such that each word vector is multiplied by its tfidf score
    vec_seq = []
    try:
        for word in sentence.translate(None, punctuation).lower().split():
#    for word in sentence w/o punct. and cases:
            if word in word2vec_mod and word in tfidf_dic:
                vec_seq.append(word2vec_mod[word] * tfidf_dic[word])
    except TypeError:
        print("saw float in: " + str(sentence))
    return vec_seq


data = pd.read_csv(sys.argv[1])
output = []
trs = 'train' in sys.argv[1]
tfidf_dic = build_tfidf_dic(list(data['question1']) + list(data['question2']))
for sentences in zip(data['question1'], data['question2']):
    nr = [get_words(sentences[0], tfidf_dic), get_words(sentences[1], tfidf_dic)]
    output.append(nr)

np.save(sys.argv[1].split('.')[0] + '.npy', np.asarray(output))
if trs:
    np.save(sys.argv[1].split('.')[0] + 'labels.npy', np.asarray(data['is_duplicate']))
