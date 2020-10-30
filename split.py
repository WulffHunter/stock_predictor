from numpy import array
from pickle import dump

import pandas as pd

# # load a dataset
# def load_file(filename):
#     dat_binary = []
#     with open(input_file_name, "r") as f:
#         line_cnt = 0
#         for line in f:
#             curr_val = line.rstrip("\n")
#             dat_binary.append(curr_val)
#             line_cnt = line_cnt + 1

#     return dat_binary

def load_csv(filename):
    datafile = pd.read_csv(filename)
    
    return datafile.close.tolist()


# save a list of sequences to file as a pkl
def save_sequence(sequence, filename):
    dump(sequence, open(filename, 'wb'))
    print('Saved: %s' % filename)

# Splits a sequence into samples. Each sample is `step_count` elements in length.
# The X is the sample, the y is the value that comes after the i-th sample
def split_into_samples(sequence, step_count):
    X_y = []

    for i in range(len(sequence)):
        # Find the end of this particular pattern (e.g. 5 elements per step)
        end_index = i + step_count

        if end_index > len(sequence) - 1:
            break

        seq_x, seq_y = sequence[i:end_index], sequence[end_index]

        X_y.append(array( seq_x, seq_y ))

    print(X_y)
    return array(X_y)

def main(in_file: "Sequence input in non-binary format (e.g. CSV)",
         out_file: "Out files title (X_y file will be saved as ____.pkl)",
         step_count: ("The number of steps per sub-sequence", 'option', 's') = 5):
        
    # sequence = load_file(in_file)
    sequence = load_csv(in_file)

    X_y = split_into_samples(sequence=sequence, step_count=step_count)

    save_sequence(X_y, out_file + '.pkl')

if __name__ == "__main__":
    import plac

    plac.call(main)
