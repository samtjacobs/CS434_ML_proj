#This is a basic model that will be used with word embeddings for question semantic similarity predictions

import numpy as np
from keras.models import Sequential, Model
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Dropout, Input, Lambda, LSTM, merge
from keras.layers.core import Masking
from keras.optimizers import RMSprop
from keras.layers.merge import Dot

maxlen = 30

def load_training_data():
    trs = np.load('../preproc/train.npy')
    gt = np.load('../preproc/labels.npy')
    assert len(trs) == len(gt)
    return trs, gt


#using TensorFlow backend
def create_LSTM():
    seq = Sequential()
    # add masking layer to ignore empty words in sentence list
    #seq.add(Masking(mask_value=0., input_shape=(timesteps, features)))
    #need to decide on LSTM arguments still, probably most will be default, except maybe activation would be better relu than linear?

    seq.add(LSTM(32, input_shape=(maxlen,300)))
    seq.add(Dense(32))
    #also need to decide whether to add more LSTM layers, dense layers, how many... not sure what the intuition for this is to narrow choices,
    #so that I can then start testing effects on prediction accuracy

    return seq

def main():
    #load data into array in the form:
    #train_data is in an array consisting of question one word embeddings, question two word embeddings, duplicate question label
    #ditto for test_data
    # TODO
    # Load in q1_data and q2_data they are embeddings generated by preprocessing
    trs, gt = load_training_data()
    q1, q2 = trs[:,0], trs[:,1]
    # Pads lists of embeddings
    # Now, each question in the training set is an array of 30*300, where each word is 300 features
    # Each question is 30 words.  If the real question is less than 30 words, it is prependedd with
    # empty words which are arrays of 300 zeros
    # The input_shape of the first layer LSTM should reflect this.
    # The word embeddings should now be usable as training data
    q1_word_embeddings = pad_sequences(q1, maxlen=maxlen, dtype='float32',padding='pre', truncating='pre', value=0.)
    q2_word_embeddings = pad_sequences(q2, maxlen=maxlen, dtype='float32',padding='pre', truncating='pre', value=0.)
    #create base LSTM models
    sibling_LSTM = create_LSTM()

    #I don't know until later today how my partner's word embeddings are formatted, and how to give them to this input layer, and whether dimensions should be specified
    #q1_word_embeddings = Input()#specify input shape as parameter
    #q2_word_embeddings = Input()
    first_question = sibling_LSTM()
    second_question = sibling_LSTM()

    merger = Dot(normalize=True)
    model = merge(input=[first_question, second_question], output=merger)#cosine similarity for now, would like to try KL-divergence later

    # train

    #compile with mean squared error and use RMSprop (generally good for recurrent networks)
    model.compile(loss='mean_squared_error', optimizer='RMSprop')

    #again the formatting of the data will be important here, and parameters will change, but I'm just looking for something simple to do parameter testing with
    model.fit([q1_data, q2_data], labels, validation_split=.20,
              batch_size=100, verbose=2, nb_epoch=10)

    # compute final accuracy on training and test sets
    training_predictions = model.predict(train_data)
    test_predictions = model.predict(test_data)

main()