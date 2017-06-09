#This is a basic model that will be used with word embeddings for question semantic similarity predictions

import numpy as np
import h5py as h5py
from keras.models import Sequential, Model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.layers import Dense, Dropout, Input, Lambda, LSTM, merge, Merge
from keras.optimizers import RMSprop
from keras.layers.merge import Dot
from keras import backend as K
import pandas as pd
import csv

maxlen = 30
MAX_NB_WORDS = 200000

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
    tokens = make_tokens(question1, question2)
    # Make tokens of training data
    # tokens.word_index is a dictionary, given word as key
    # will return frequency based on index
    # Pre-trained
    input_q1 = Input(shape=(maxlen,300),dtype='float32',name='q1')
    input_q2 = Input(shape=(maxlen,300),dtype='float32',name='q2')
    siamese_LSTM = create_LSTM((maxlen,300))

    question_1 = siamese_LSTM(input_q1)
    question_2 = siamese_LSTM(input_q2)
    distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([question_1, question_2])

    model = Model(input=[input_q1, input_q2], output=distance)
    #compile with mean squared error and use RMSprop (generally good for recurrent networks)
    model.compile(loss='mean_squared_error', optimizer='RMSprop')

    #again the formatting of the data will be important here, and parameters will change, but I'm just looking for something simple to do parameter testing with
    #model.fit([q1_word_embeddings, q2_word_embeddings], gt)
    model.fit([q1_word_embeddings, q2_word_embeddings], gt, validation_split=.20,
              batch_size=100, verbose=2, nb_epoch=10)
    model.save('quora_regressor.h5')
    # compute final accuracy on training and test sets

main()