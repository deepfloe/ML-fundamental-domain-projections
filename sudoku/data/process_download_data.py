import numpy as np
from os.path import exists
from zipfile import ZipFile


def dataset_line_to_list(line):
    return [int(x) for x in line[13:-6]]

def generate_dataset(train_size = 1000, test_size = 5000):
    SOURCE_FILES = [
        'raw/easy.txt', 'raw/medium.txt', 'raw/hard.txt', 'raw/diabolical.txt'
    ]
    train = []
    test = []
    for filename in SOURCE_FILES:
        with open(filename) as f:
            for num, line in enumerate(f):
                if num < train_size/4:
                    train += [dataset_line_to_list(line)]
                elif num >= train_size/4 and num-train_size/4 < test_size/4:
                    test += [dataset_line_to_list(line)]
                else:
                    break
    np.save('processed/train.npy', np.array(train))
    np.save('processed/test.npy', np.array(test))

def verify_dataset(train_size, test_size):
    train = np.load('processed/train.npy')
    assert train.shape == (train_size, 81)
    test = np.load('processed/test.npy')
    assert test.shape == (test_size, 81)


def main():
    TRAIN_SIZE = 1000
    TEST_SIZE = 5000
    generate_dataset(train_size=TRAIN_SIZE, test_size=TEST_SIZE)
    verify_dataset(TRAIN_SIZE, TEST_SIZE)


if __name__ == '__main__':
    main()
