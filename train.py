from __future__ import print_function

import sys
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, Callback

# from model import get_model
from model import simple_model
# from utils import crps, real_to_cdf, preprocess, rotation_augmentation, shift_augmentation

from collections import namedtuple

import gc
import os
import click

def split_data(X, y, split_ratio=0.2):
    """
    Split data into training and testing.

    :param X: X
    :param y: y
    :param split_ratio: split ratio for train and test data
    """
    split = X.shape[0] * split_ratio
    X_test = X[:split, :, :, :]
    y_test = y[:split, :]
    X_train = X[split:, :, :, :]
    y_train = y[split:, :]

    return X_train, y_train, X_test, y_test


def train_hard(config):
    print('-' * 50)
    print('Load data')

    X = np.load(config.x_train).astype(np.float32, copy=False)
    X /= 255.0

    y = np.load(config.y_train)

    if config.weighted:
        weights = []
        with open(config.meta_train, 'r') as fin:
            fin.readline()
            for line in fin:
                s = line.replace('\n','').split(',')
                w = float(s[2])
                weights.append(w)
        weights = np.array(weights).reshape(-1, 1)
        y = np.hstack([y, weights])
    print('Done.')
    print('Shuffle and split')

    np.random.seed(config.seed)
    np.random.shuffle(X)
    np.random.seed(config.seed)
    np.random.shuffle(y)

    X_train, y_train, X_test, y_test = split_data(X, y, split_ratio=0.2)

    gc.collect()
    print('Load model')
    # model = w_simple_model()
    model = simple_model()

    nb_iter = 200
    epochs_per_iter = 1
    batch_size = 32

    min_val = sys.float_info.max

    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=45,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=True)  # randomly flip images

    print('-'*50)
    print('Training...')
    print('-'*50)

    datagen.fit(X_train)

    checkpointer_best = ModelCheckpoint(filepath=config.weights_prefix + 'weights_best.hdf5', verbose=1, save_best_only=True)
    checkpointer = ModelCheckpoint(filepath=config.weights_prefix + 'weights.hdf5', verbose=1, save_best_only=False)

    def weight_wrapper(data_flow):
        for X_batch, y_batch in data_flow:
            yield X_batch, y_batch[:, 0], y_batch[:, 1]

    if config.weighted:
        dataFlow = weight_wrapper(datagen.flow(X_train, 
                                               y_train[:, [config.col, -1]],
                                               batch_size=batch_size))
    else:
        dataFlow = datagen.flow(X_train, y_train[:, config.col],
                                batch_size=batch_size)


    hist = model.fit_generator(dataFlow,
                               samples_per_epoch=X_train.shape[0],
                               nb_epoch=nb_iter, show_accuracy=False,
                               validation_data=(X_test, y_test[:, config.col]),
                               callbacks=[checkpointer, checkpointer_best],
                               nb_worker=config.nb)

    with open(config.weights_prefix + 'val_loss.txt', mode='w+') as f:
        f.write(str(min(hist.history['val_loss'])))

    print('Make train predict')
    pred = model.predict(X, batch_size=batch_size, verbose=1)
    np.save(config.pred_prefix + 'y-train.npy', pred)

    X = np.load(config.x_test).astype(np.float32, copy=False)
    X /= 255.0

    print('Make test predict')
    pred = model.predict(X, batch_size=batch_size, verbose=1)
    np.save(config.pred_prefix + 'y-test.ny', pred)


def new_run(method=None):
    Config = namedtuple('Config', ['seed', 'weighted', 'x_train', 'y_train', 'meta_train', 'col',
                                   'nb', 'weights_prefix', 'pred_prefix', 'x_test'])
    seed = 19595

    # run without weighting

    if method is None or method == 'mask':
        npy_path = '/data/backup/mar8/fft-mask-w-multiscale-'
    if method == 'h1mask':
        npy_path = '/data/backup/mar8/fft-h1mask-w-multiscale-'
    if method == 'h1attend':
        npy_path = '/data/backup/mar8/fft-h1attend-w-multiscale-'

    config = Config(seed,
                    False,
                    npy_path + 'X-train.npy',
                    npy_path + 'y-train.npy',
                    npy_path + 'meta-train.csv',
                    0,
                    8,
                    npy_path + 'systole-',
                    npy_path + 'systole-pred-',
                    npy_path + 'X-test.npy')
    train_hard(config)
    gc.collect()

    config = Config(seed,
                    False,
                    npy_path + 'X-train.npy',
                    npy_path + 'y-train.npy',
                    npy_path + 'meta-train.csv',
                    1,
                    8,
                    npy_path + 'diastole-',
                    npy_path + 'diastole-pred-',
                    npy_path + 'X-test.npy')
    train_hard(config)


@click.command()
@click.option('--method', default='mask')
def main(method):
    if not method in ['mask', 'h1mask', 'h1attend']:
        method = None
    new_run(method)


if __name__ == "__main__":
    main()
