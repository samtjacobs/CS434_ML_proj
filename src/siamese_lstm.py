#This is a basic model that will be used with word embeddings for question semantic similarity predictions

import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, Lambda
from keras.optimizers import RMSprop
from keras.layers.merge import Dot

#using TensorFlow backend

def create_LSTM():
    seq = Sequential()

    #need to decide on LSTM arguments still, probably most will be default, except maybe activation would be better relu than linear?

    seq.add(LSTM())
    #seq.add(Dense(100))
    #also need to decide whether to add more LSTM layers, dense layers, how many... not sure what the intuition for this is to narrow choices,
    #so that I can then start testing effects on prediction accuracy

    return seq

def main():
    #load data into array in the form:
    #question one word embeddings, question two word embeddings, duplicate question label
    #TODO

    #create base LSTM models
    sibling_LSTM = create_LSTM()

    #I don't know until later today how my partner's word embeddings are formatted, and how to give them to this input layer, and whether dimensions should be specified
    q1_word_embeddings = Input()#specify input shape as parameter
    q2_word_embeddings = Input()
    first_question = sibling_LSTM(q1_data)
    second_question = sibling_LSTM(q2_data)

    merger = Dot(normalize=True)
    model = merge(input=[first_question, second_question], output=merger)#cosine similarity for now, would like to try KL-divergence later

    # train

    #compile with mean squared error and use RMSprop (generally good for recurrent networks)
    model.compile(loss='mean_squared_error', optimizer='RMSprop')

    #again the formatting of the data will be important here, and parameters will change, but I'm just looking for something simple to do parameter testing with
    model.fit([q1_data, q2_data], labels, validation_split=.20,
              batch_size=100, verbose=2, nb_epoch=10)

    # compute final accuracy on training and test sets
    training_predictions = model.predict(data)
    test_predictions = model.predict(test_data)

main()