import numpy as np
from pickle import load


def load_sequence(filename):
    return load(open(filename, 'rb'))

def get_X_y(filename):
    X_y = load_sequence(filename)

    X = np.vstack(X_y[:, 0])
    # Fix some Tensorflow errors
    X = np.asarray(X).astype('float32')

    y = np.array(X_y[:, 1])
    y.shape = (y.size, 1)
    # Fix some Tensorflow errors
    y = np.asarray(y).astype('float32')

    return X, y