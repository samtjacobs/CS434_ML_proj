#This is a basic model that will be used with word embeddings for question semantic similarity predictions

import numpy as np
import h5py as h5py
from keras.models import Sequential, Model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.layers import Dense, Dropout, Input, Lambda, LSTM, merge, Merge
from keras.optimizers import RMSprop
from keras.layers import Embedding
from keras import backend as K
import pandas as pd
import csv
import cPickle as pickle
import sys

maxlen = 30
MAX_NB_WORDS = 200000
PREPROC_PATH = '../preproc'
DATA = sys.argv[1]
EMBEDDING_DIM = 300
MAX_SEQUENCE_LENGTH = 64

def get_q_strings(data_path):
    question1 = []
    question2 = []
    with open(data_path) as csvfile:
        reader = csv.DictReader(csvfile, delimiter=',')
        for row in reader:
            question1.append(row['question1'])
            question2.append(row['question2'])
    return question1, question2


def make_tokens(question1, question2):
    questions = question1 + question2
    tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(questions)
    return tokenizer


def get_embed_weights(word_ind, embed_ind):
    embedding_matrix = np.zeros((len(word_ind) + 1, EMBEDDING_DIM))
    for word, i in word_ind.items():
        embedding_vector = embed_ind.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    return embedding_matrix


def make_embed_layer(word_ind, embed_w):
    return Embedding(len(word_ind) + 1,
                    EMBEDDING_DIM,
                    weights=[embed_w],
                    input_length=MAX_SEQUENCE_LENGTH,
                    trainable=False)


#using TensorFlow backend
def create_LSTM(input):
    seq = Sequential()
    # add masking layer to ignore empty words in sentence list
    #seq.add(Masking(mask_value=0., input_shape=(timesteps, features)))
    #need to decide on LSTM arguments still, probably most will be default, except maybe activation would be better relu than linear?

    seq.add(LSTM(32, input_shape=input))
    seq.add(Dense(1))
    #also need to decide whether to add more LSTM layers, dense layers, how many... not sure what the intuition for this is to narrow choices,
    #so that I can then start testing effects on prediction accuracy

    return seq

def main():
    question1, question2 = get_q_strings('../train.csv')
    word_ind = pickle.load(PREPROC_PATH + 'word_ind.p')
    embedding_ind = pickle.load(PREPROC_PATH + 'word_vec.p')
    # Make tokens of training data
    # tokens.word_index is a dictionary, given word as key
    # will return frequency based on index
    # Pre-trained
    embed_w = get_embed_weights(word_ind, embedding_ind)

    input_q1 = Input(shape=(maxlen,300),dtype='float32',name='q1')
    input_q2 = Input(shape=(maxlen,300),dtype='float32',name='q2')
    siamese_LSTM = create_LSTM((maxlen,300))

if __name__ == "__main__":
    main()