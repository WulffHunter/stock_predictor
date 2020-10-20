from pickle import load


def load_sequence(filename):
    return load(open(filename, 'rb'))