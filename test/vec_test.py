from keras.preprocessing.sequence import pad_sequences
import numpy as np
'''Should be ignored'''
seq1 = np.array([[1,1,1],[2,2,2],[3,3,3]])
seq2 = np.array([[2,2,2],[3,3,3]])
seq3 = np.array([[3,3,3]])
seqs = [seq1, seq2, seq3]

padded = pad_sequences(seqs, maxlen=None, dtype='int32',padding='pre', truncating='pre', value=0.)

print(str(seqs))