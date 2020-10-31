'''
Split the dataset into train and test
'''
from pickle import dump

import numpy as np
from numpy.random import shuffle

from utils import load_sequence

# save a list of clean sequences to file
def save_clean_data(sequences, filename):
    dump(sequences, open(filename, 'wb'))
    print('Saved: %s' % filename)


def main(in_file: "Input raw file in numpy binary format",
         train_set_percentage: ("Percent of the observations to use for training", 'option', 'p') = 90,
         max_observations: ("The number of observations to read (and process) from file", 'option', 'm') = None):
    rnd_seed = 12345

    train_dat_file_name = in_file + '.train.pkl'
    test_dat_file_name = in_file + '.test.pkl'

    print(f'The random seed is set to {rnd_seed}.')
    print(f'Allocate {train_set_percentage}% of observations for train.')
    print(f'Store train dataset in {train_dat_file_name}')
    print(f'Store test dataset in {test_dat_file_name}')

    # load dataset
    raw_dataset = load_sequence(in_file)
    # reduce dataset size
    number_of_rows = np.size(raw_dataset, 0)

    if max_observations is None:
        max_observations = number_of_rows

    if number_of_rows < max_observations:
        raise ValueError(f'The number of observations is {number_of_rows}, which is smaller than '
                         f'the maximum number of observations (set to {max_observations}). '
                         f'Reduce the max_observations value.')
    n_sequences = max_observations
    dataset = raw_dataset[:n_sequences, :]
    print(dataset)
    # random shuffle
    # note that we do not need it for the toy example (based on how we currently generate the data)
    # but let's keep it here for completeness
    print(f'Reshuffle the data')
    np.random.default_rng(rnd_seed).shuffle(dataset)
    # split into train/test
    train_set_size = int(n_sequences * train_set_percentage / 100.0)
    test_set_size = n_sequences - train_set_size
    print(f'Save {train_set_size} pairs for train and {test_set_size} pairs for test.')

    train, test = dataset[:train_set_size], dataset[train_set_size:]
    # save
    save_clean_data(train, train_dat_file_name)
    save_clean_data(test, test_dat_file_name)


if __name__ == "__main__":
    import plac

    plac.call(main)
