#This is a basic model that will be used with word embeddings for question semantic similarity predictions

import numpy as np
import h5py as h5py
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils.np_utils import to_categorical
from keras.layers import Dense, Dropout, Input, Lambda, TimeDistributed, LSTM, BatchNormalization, concatenate, Conv1D
from keras.optimizers import Adam
from keras.layers import Embedding
from keras import backend as K
from keras.models import Model, Sequential
import csv
import cPickle as pickle
import time
from keras.callbacks import Callback, ModelCheckpoint


maxlen = 30
MAX_NB_WORDS = 200000
PREPROC_PATH = '../preproc/'
EMBEDDING_DIM = 300
HIDDEN_DIM = 100
DROPOUT = 0.2
MAX_SEQUENCE_LENGTH = 64
VALIDATION_SPLIT = 0.90
MNAME = 'quora_mod'

def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))

def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)


def get_q_strings(data_path):
    question1 = []
    question2 = []
    gt = []
    with open(data_path) as csvfile:
        reader = csv.DictReader(csvfile, delimiter=',')
        for line in reader:
            question1.append(line['question1'])
            question2.append(line['question2'])
            gt.append(line['is_duplicate'])
    return question1, question2, gt


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

def make_shared_leg(input_dim, word_ind, embed_w):
    inp = Input(shape=(input_dim, ))
    leg = make_embed_layer(word_ind, embed_w)(inp)
    leg = LSTM(EMBEDDING_DIM, activation='relu', dropout=0.2, recurrent_dropout=0.2)(leg)
    leg = Dense(EMBEDDING_DIM, activation='relu')(leg)
    leg = Dropout(DROPOUT)(leg)
    leg = BatchNormalization()(leg)
    #leg = Lambda(lambda x: K.max(x, axis=1), output_shape=(EMBEDDING_DIM,))(leg)
    return Model(input=inp, output=leg)

def main():
    word_ind = pickle.load(open(PREPROC_PATH + 'word_ind.p'))
    embedding_ind = pickle.load(open(PREPROC_PATH + '6B_word_vec.p'))
    q_seqs = pickle.load(open(PREPROC_PATH + 'que_seqs.p'))
    gt = pickle.load(open(PREPROC_PATH + 'labels.p'))
    q1_seq = q_seqs[0]
    q2_seq = q_seqs[1]
    q1_data = pad_sequences(q1_seq, maxlen=MAX_SEQUENCE_LENGTH)
    q2_data = pad_sequences(q2_seq, maxlen=MAX_SEQUENCE_LENGTH)

    #labels = to_categorical(np.asarray(gt, dtype=int))
    labels = np.asarray(gt, dtype='float32')
 #  indices = np.arange(q1_data.shape[0])
 #   np.random.shuffle(indices)
 #   q1_data = q1_data[indices]
 #   q2_data = q2_data[indices]
 #   labels = labels[indices]
    nb_validation_samples = int(VALIDATION_SPLIT * q1_data.shape[0])

    # Seperate testing and validation sets
    q1_train = q1_data[:-nb_validation_samples]
    q2_train = q2_data[:-nb_validation_samples]
    l_train = labels[:-nb_validation_samples]
    q1_val = q1_data[-nb_validation_samples:]
    q2_val = q2_data[-nb_validation_samples:]
    l_val = labels[-nb_validation_samples:]

    # Make model.  Siamese.  Two legs, one for each question.  Share embedding
# and LSTM layers
# Note: I am using the Keras functional API, not sequential or graph
    embed_w = get_embed_weights(word_ind, embedding_ind)
    q1 = Input(shape=(MAX_SEQUENCE_LENGTH, ))
    q2 = Input(shape=(MAX_SEQUENCE_LENGTH, ))
    # Might add mask
    # TimeDistributed is 2D LSTM from my understanding
    shared = make_shared_leg(MAX_SEQUENCE_LENGTH, word_ind, embed_w)
   # print(type(shared))
    q1_path = shared(q1)
    q2_path = shared(q2)

    merged = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([q1_path, q2_path])
    # Pass thru single path
    merged = Dense(200, activation='relu')(merged)
    merged = Dropout(DROPOUT)(merged)
    merged = BatchNormalization()(merged)
    merged = Dense(100, activation='relu')(merged)
    merged = Dropout(DROPOUT)(merged)
    merged = BatchNormalization()(merged)
    # Final
    pred = Dense(1, activation='sigmoid')(merged)

    model = Model(inputs=[q1, q2], output=pred)
    TIME = str(time.localtime().tm_hour) + str(time.localtime().tm_min)
    model.compile(loss='mse', optimizer='adam', metrics=['acc'])
    checkpoint = [ModelCheckpoint(MNAME + '_' + TIME + '_weights.h5', monitor='val_acc', save_best_only=True, verbose=2)]
    history = model.fit([q1_train, q2_train], l_train,
              validation_data=([q1_val, q2_val], l_val),
              epochs=10, batch_size=128, verbose=2, callbacks=checkpoint)

    max_val_acc, idx = max((val, idx) for (idx, val) in enumerate(history.history['val_acc']))
    print('Maximum validation accuracy = {0:.4f} (epoch {1:d})'.format(max_val_acc, idx+1))
    model.load_weights(MNAME + '_' + TIME +'_weights.h5')
    predictions = model.predict([q1_data, q2_data], batch_size=100, verbose=0)
    np.save(open(TIME + '_' + "predictions.np", "wb"), predictions)
    model.save(MNAME + '_' + TIME + '.h5')

if __name__ == "__main__":
    main()