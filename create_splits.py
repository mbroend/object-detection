import argparse
import glob
import os
import random
import tensorflow as tf

import numpy as np

from utils import get_module_logger, get_dataset
import shutil
import random


def split(source, destination):
    """
    Create three splits from the processed records. The files should be moved to new folders in the
    same directory. This folder should be named train, val and test.

    args:
        - source [str]: source data directory, contains the processed tf records
        - destination [str]: destination data directory, contains 3 sub folders: train / val / test
    """
    dir_list = os.listdir(source)
    
    dataset_size = len(dir_list)
    # Test and val size - train will be remainder
    test_size = 3
    val_size = int(0.2 * (dataset_size-test_size))

    # Hide away a few files for testing afterwards
    test_files = random.sample(dir_list, k=test_size)

    for item in test_files:
        dir_list.remove(item)

    val_files = random.sample(dir_list, k=val_size)
    for item in val_files:
        print(item)
        dir_list.remove(item)
    train_files = dir_list

    print(f"Number of test files: {len(test_files)}")
    print(f"Number of train files: {len(train_files)}")
    print(f"Number of val files: {len(val_files)}")

    print("Copying files...")
    for elem in test_files:
        shutil.copy(source+elem,destination+"/test/"+elem)
    for elem in val_files:
        shutil.copy(source+elem,destination+"/val/"+elem)
    for elem in train_files:
        shutil.copy(source+elem,destination+"/train/"+elem)


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