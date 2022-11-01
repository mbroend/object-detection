import argparse
import glob
import os
import random
import tensorflow as tf

import numpy as np

from utils import get_module_logger, get_dataset


def split(source, destination):
    """
    Create three splits from the processed records. The files should be moved to new folders in the
    same directory. This folder should be named train, val and test.

    args:
        - source [str]: source data directory, contains the processed tf records
        - destination [str]: destination data directory, contains 3 sub folders: train / val / test
    """
    # TODO: Implement function
    

    count = 0
    # Iterate directory
    for path in os.listdir(source):
        # check if current path is a file
        if os.path.isfile(os.path.join(source, path)):
            count += 1
    print('File count:', count)



    dataset = get_dataset(f"{source}*.tfrecord")
    dataset = dataset.shuffle(100)
    #dataset_size = len(list(dataset))
    dataset_size = 100
    print(dataset_size)
    test_size = 3
    train_size = int(0.8 * dataset_size-test_size)
    val_size = int(0.2 * dataset_size-test_size)
    

    test_dataset = dataset.take(test_size)
    train_dataset = dataset.skip(test_size)
    val_dataset = dataset.skip(val_size)
    train_dataset = dataset.take(train_size)

    print('her')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Split data into training / validation / testing')
    parser.add_argument('--source', required=True,
                        help='source data directory')
    parser.add_argument('--destination', required=True,
                        help='destination data directory')
    args = parser.parse_args()

    logger = get_module_logger(__name__)
    logger.info('Creating splits...')
    split(args.source, args.destination)