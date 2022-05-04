import numpy as np
from os.path import exists
from zipfile import ZipFile


def dataset_line_to_X_y_formatted(line):
    return [int(x) for x in line[13:-6]], float(line[-4:])

def generate_dataset(size = 1000):
    SOURCE_FILES = [
        'raw/easy.txt', 'raw/medium.txt', 'raw/hard.txt', 'raw/diabolical.txt'
    ]
    X = []
    y = []
    for filename in SOURCE_FILES:
        with open(filename) as f:
            for num, line in enumerate(f):
                if num < size/4:
                    X += [dataset_line_to_X_y_formatted(line)[0]]
                    y += [dataset_line_to_X_y_formatted(line)[1]]
                else:
                    break
    np.save('processed/X.npy', np.array(X))
    np.save('processed/y.npy', np.array(y))

def verify_dataset(size):
    X = np.load('processed/X.npy')
    assert X.shape == (size, 81)
    y = np.load('processed/y.npy')
    assert y.shape == (size,)


def main():
    SIZE = 300000
    generate_dataset(size = SIZE)
    verify_dataset(SIZE)


if __name__ == '__main__':
    main()
