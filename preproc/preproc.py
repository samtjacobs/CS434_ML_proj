from keras.preprocessing.text import Tokenizer
import csv
import sys
import os
import numpy as np
import cPickle as pickle
MAX_NB_WORDS = 200000

def get_q_strings(data_path):
    question1 = []
    question2 = []
    labels = []
    with open(data_path) as csvfile:
        reader = csv.DictReader(csvfile, delimiter=',')
        for row in reader:
            question1.append(row['question1'])
            question2.append(row['question2'])
            labels.append(int(row['is_duplicate']))
    return question1, question2, labels


def make_tokens(question1, question2):
    questions = question1 + question2
    tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(questions)
    return tokenizer


def get_embeddings(data_path):
    embeddings_index = {}
    f = open(data_path)
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    return embeddings_index


def main():
    data_path = sys.argv[1]
    q1, q2, gt = get_q_strings(data_path)
    tokens = make_tokens(q1, q2)
    pickle.dump(tokens.word_index, open("word_ind.p", "wb"))
    pickle.dump(gt, open("labels.p", "wb"))
    embed_dic = get_embeddings(sys.argv[2])
    pickle.dump(embed_dic, open(sys.argv[2].split('.')[1] + "_word_vec.p", "wb"))
    seqs = [tokens.texts_to_sequences(q1), tokens.texts_to_sequences(q2)]
    pickle.dump(seqs, open("que_seqs.p", "wb"))


if __name__ == "__main__":
    main()