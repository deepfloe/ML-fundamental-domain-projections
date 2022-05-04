from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Conv2D, AveragePooling2D, Flatten
import numpy as np
import pickle
import tensorflow as tf
from sudoku.data.load_data import load_data
from sklearn.model_selection import train_test_split


def get_network(layerList):
    inp = tf.keras.layers.Input(shape=layerList.pop(0)[0])
    layers = [tf.keras.layers.Reshape((-1,))(inp)]

    for k in range(len(layerList)):
        layer_data = layerList.pop(0)
        layers += [tf.keras.layers.Dense(layer_data[0], activation=layer_data[1])(layers[k])]

    model = tf.keras.models.Model(inputs=inp, outputs=layers[-1])
    model.compile(
        loss='mean_squared_error',
        optimizer=tf.keras.optimizers.Adam(0.001),
        metrics=['MeanSquaredError'],
    )
    return model

def get_medium_nn():
    arch = [
        ((81, ), 'ignored'),
        (256, 'relu'),
        (128, 'relu'),
        (64, 'relu'),
        (32, 'relu'),
        (1, 'relu')
    ]
    return get_network(arch)

def get_medium_with_dropout():
    inp = tf.keras.layers.Input(shape=(81,))
    l1 = tf.keras.layers.Dense(256, activation='relu')(inp)
    l1d = tf.keras.layers.Dropout(0.5)(l1)
    l2 = tf.keras.layers.Dense(128, activation='relu')(l1d)
    l2d = tf.keras.layers.Dropout(0.5)(l2)
    l3 = tf.keras.layers.Dense(64, activation='relu')(l2d)
    l3d = tf.keras.layers.Dropout(0.5)(l3)
    l4 = tf.keras.layers.Dense(32, activation='relu')(l3d)
    l5 = tf.keras.layers.Dense(1, activation='relu')(l4)

    model = tf.keras.models.Model(inputs=inp, outputs=l5)
    model.compile(
        loss='mean_squared_error',
        optimizer=tf.keras.optimizers.Adam(0.001),
        metrics=['MeanSquaredError'],
    )
    return model

def main():
    print('Loading data...')
    X, y = load_data()
    print('Data loaded.')
    print('Splitting data...')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 1)
    print('Data splitted.')
    model = get_medium_with_dropout()
    print('Training...')
    hist = model.fit(
        X_train, y_train,
        epochs=100,
        validation_data=(X_test, y_test),
    )
    print('Training ended.')
    print(hist.history['val_mean_squared_error'][-1])


if __name__ == '__main__':
    main()