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
         val_set_percentage: ("Percent of the observations to use for validation", 'option', 'v') = None,
         test_set_percentage: ("Percent of the observations to use for training", 'option', 't') = None,
         max_observations: ("The number of observations to read (and process) from file", 'option', 'm') = None):
    rnd_seed = 12345

    train_dat_file_name = in_file + '.train.pkl'
    val_dat_file_name = in_file + '.val.pkl'
    test_dat_file_name = in_file + '.test.pkl'

    print(f'The random seed is set to {rnd_seed}.')
    print(f'Allocate {val_set_percentage}% of observations for validation.')
    print(f'Allocate {test_set_percentage}% of observations for test.')
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
    
    # split into train/validate/test
    val_set_size = 0 if val_set_percentage is None else int(n_sequences * float(val_set_percentage) / 100.0)
    test_set_size = 0 if test_set_percentage is None else int(n_sequences * float(test_set_percentage) / 100.0)
    train_set_size = (n_sequences - val_set_size) - test_set_size
    
    print(f'Save {train_set_size} entries for train, {val_set_size} entries for validation, and {test_set_size} entries for test.')

    train_val_set = dataset[:n_sequences - test_set_size]
    # Choose the last few elements of the set, unshuffled
    test = dataset[n_sequences - test_set_size:]

    # random shuffle the train and validation sets
    # note that we do not need it for the toy example (based on how we currently generate the data)
    # but let's keep it here for completeness
    print(f'Reshuffle the training and validation data')
    np.random.default_rng(rnd_seed).shuffle(train_val_set)

    train = train_val_set[:train_set_size]
    val = train_val_set[train_set_size:]

    save_clean_data(train, train_dat_file_name)

    # save
    if val_set_percentage is not None:
        save_clean_data(val, val_dat_file_name)

    if test_set_percentage is not None:
        save_clean_data(test, test_dat_file_name)


if __name__ == "__main__":
    import plac

    plac.call(main)
