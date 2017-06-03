import keras as K
import numpy as np
from keras.models import Sequential

def load():
    trs = np.load('../preproc/train.npy')
    gt = np.load('../preproc/labels.npy')
    assert len(trs) == len(gt)
    return trs, gt


def create_base_network():
    '''Create Base Network which is to be
        duplicted as floating layer to form
        Siamese network'''
    seq = Sequential()
    # TODO research best parameters for network
    return seq
def main():
    trs, gt = load()

if __name__ == "__main__":
    main()