import numpy as np
from tqdm import tqdm
from sudoku.ordmaps.dirichlet.discrete_gradient_ascent import dirichlet_ord


def main():
    X = np.load('processed/X.npy')
    assert X.shape[1] == 81
    X_new = []
    for mat in tqdm(X):
        X_new += [dirichlet_ord(np.reshape(mat, (9,9,)))]
    np.save('processed/X_dirichlet_ord.npy', np.array(X_new))


if __name__ == '__main__':
    main()
