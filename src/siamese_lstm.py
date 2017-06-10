#This is a basic model that will be used with word embeddings for question semantic similarity predictions

import numpy as np
from keras.models import Sequential, Model
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Dropout, Input, Lambda, LSTM, merge, Merge
from keras.optimizers import RMSprop
from keras.layers.merge import Dot
from keras import backend as K
from keras.models import load_model

maxlen = 30

def cosine_distance(vests):
    x, y = vests
    x = K.l2_normalize(x, axis=-1)
    y = K.l2_normalize(y, axis=-1)
    return -K.mean(x * y, axis=-1, keepdims=True)

def cos_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0],1)

def load_training_data():
    trs = np.load('../preproc/train_tr.npy')
    gt = np.load('../preproc/labels.npy') #labels2 is labels for full set
    gt = gt[:len(trs)]
    assert(len(gt) == len(trs))
    return trs, gt

def get_summary(actuals, predictions):
    total_mse, acc, tp, tn, fp, fn = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    for i in range(0, len(predictions)):
        total_mse += (predictions[i] - actuals[i])**2
        if(actuals[i] == 1 and predictions[i] > 0.5):
            tp += 1
        elif(actuals[i] == 0 and predictions[i] <= 0.5):
            tn += 1
        elif(actuals[i] == 1 and predictions[i] <= 0.5):
            fn += 1
        elif(actuals[i] == 0 and predictions[i] > 0.5):
            fp += 1
    acc = (tp + tn) / len(predictions)
    assert(tp + tn + fn + fp == len(predictions))
    mse = total_mse / len(predictions)
    return mse, acc, tp, tn, fp, fn

def get_acc(predictions, actuals):
    correct = 0
    acc = 0.0
    for i in range (0, len(predictions)):
        if(-0.5 < predictions[i] - actuals[i] < 0.5):
            correct += 1
    acc = correct/len(predictions)
    return acc

#using TensorFlow backend
def create_LSTM(input):
    seq = Sequential()
    # add masking layer to ignore empty words in sentence list
    #seq.add(Masking(mask_value=0., input_shape=(timesteps, features)))
    #need to decide on LSTM arguments still, probably most will be default, except maybe activation would be better relu than linear?

    seq.add(LSTM(32, input_shape=input))
    seq.add(Dense(32))
    #also need to decide whether to add more LSTM layers, dense layers, how many... not sure what the intuition for this is to narrow choices,
    #so that I can then start testing effects on prediction accuracy

    return seq

def main():
    # #load data into array in the form:
    # #train_data is in an array consisting of question one word embeddings, question two word embeddings, duplicate question label
    # #ditto for test_data
    # # Load in q1_data and q2_data they are embeddings generated by preprocessing
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
    # input_q1 = Input(shape=(maxlen,300),dtype='float32',name='q1')
    # input_q2 = Input(shape=(maxlen,300),dtype='float32',name='q2')
    # siamese_LSTM = create_LSTM((maxlen,300))
    #
    # question_1 = siamese_LSTM(input_q1)
    # question_2 = siamese_LSTM(input_q2)
    # distance = Lambda(cosine_distance, output_shape=cos_dist_output_shape)([question_1, question_2])
    #
    # model = Model(input=[input_q1, input_q2], output=distance)
    # #compile with mean squared error and use RMSprop (generally good for recurrent networks)
    # model.compile(loss='mean_squared_error', optimizer='RMSprop')
    #
    # #again the formatting of the data will be important here, and parameters will change, but I'm just looking for something simple to do parameter testing with
    # #model.fit([q1_word_embeddings, q2_word_embeddings], gt)
    # model.fit([q1_word_embeddings, q2_word_embeddings], gt, validation_split=.20,
    #           batch_size=100, verbose=2, nb_epoch=15)
    # model.save('quora_regressor_full_15ep.h5')
    model = load_model('quora_regressor_100ep.h5')
    # compute final accuracy on training and test sets
    preds = model.predict([q1_word_embeddings, q2_word_embeddings], batch_size=1, verbose=1)
    mse, acc, tp, tn, fp, fn = get_summary(gt, preds)
    print("finished")#this line exists so that we can view the variables in the line above it

main()