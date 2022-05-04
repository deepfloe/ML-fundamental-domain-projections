import numpy as np
import os
import pathlib


def load_data():
    X_path = os.path.join(pathlib.Path(__file__).parent.absolute(), 'processed/X.npy')
    y_path = os.path.join(pathlib.Path(__file__).parent.absolute(), 'processed/y.npy')
    X = np.load(X_path)
    y = np.load(y_path)
    return X, y