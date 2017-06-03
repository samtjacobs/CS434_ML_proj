#This is a basic model that will be used with word embeddings for question semantic similarity predictions

import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, Lambda
from keras.optimizers import RMSprop

#using TensorFlow backend

def create_LSTM():
    seq = Sequential()

    #need to decide on LSTM arguments still, probably most will be default, except maybe activation would be better relu than linear?
    seq.add(LSTM())
    #also need to decide whether to add more LSTM layers, dense layers, how many... not sure what the intuition for this is to narrow choices,
    #so that I can then start testing effects on prediction accuracy

def main():
    #load data
    #though I don't know yet how my partner has formatted the outputs of the word embeddings,
    #so this is another issue that I'll have to discuss with him when we meet later today

    base_network = create_LSTM()

    #I don't know until later today how my partner's word embeddings are formatted, and how to give them to this input layer, and whether dimensions should be specified
    q1_word_embeddings = Input()
    q2_word_embeddings = Input()
    first_question = base_network(q1_word_embeddings)
    second_question = base_network(q2_word_embeddings)

    model = merge(input=[input_a, input_b], output=cos)#would like to try KL-divergence later

    # train

    #compile with mean squared error and use RMSprop (generally good for recurrent networks)
    model.compile(loss='mse', optimizer='RMSprop')

    #again the formatting of the data will be important here, and parameters will change, but I'm just looking for something simple to do parameter testing with
    model.fit([raw_q1_data, raw_q2_data], labels, validation_split=.20,
              batch_size=100, verbose=2, nb_epoch=10)

    # compute final accuracy on training and test sets
    training_predictions = model.predict(train_data)
    test_predictions = model.predict(test_data)

main()