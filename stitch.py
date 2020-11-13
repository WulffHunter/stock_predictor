'''
Stitches multiple pkl files together into one
'''
from pickle import dump

from utils import load_sequence

import numpy as np


def main(out_file: "The final stitched dataset",
         *in_files: "The files to stitch together"):
    print(f"Stitching files: {in_files}")
    datasets = list(map(load_sequence, in_files))

    stitched = None

    for dataset in datasets:
        if stitched is None:
            stitched = dataset
        else:
            stitched = [ *stitched, *dataset ]

    # Convert to numpy array
    stitched = np.array(stitched)

    print(f'Stitched file length: {len(stitched)}')

    dump(stitched, open(out_file, 'wb'))
    print('Saving stitched files: %s' % out_file)


if __name__ == "__main__":
    import plac

    plac.call(main)
