from __future__ import print_function

import sys
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, Callback

from model import get_model
from model import w_simple_model
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

    X = np.load(config.x).astype(np.float32, copy=False)
    X /= 255.0

    y = np.load(config.y)

    if config.weighted:
        weights = []
        with open(config.meta, 'r') as fin:
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
    model = get_model()

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
        rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=True)  # randomly flip images

    print('-'*50)
    print('Training...')
    print('-'*50)

    datagen.fit(X_train)

    checkpointer_best = ModelCheckpoint(filepath=config.w_path + 'weights_best.hdf5', verbose=1, save_best_only=True)
    checkpointer = ModelCheckpoint(filepath=config.w_path + 'weights.hdf5', verbose=1, save_best_only=False)

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

    loss = hist.history['loss'][-1]
    val_loss = hist.history['val_loss'][-1]

    with open(config.w_path + 'val_loss.txt', mode='w+') as f:
        f.write(str(min(hist.history['val_loss'])))

    pred = model.predict(X, batch_size=batch_size, verbose=1)
    np.save(config.pred_path, pred)


def run_simple():
    Config = namedtuple('Config', ['seed', 'weighted', 'x', 'y', 'meta', 'col', 'nb', 'w_path', 'pred_path'])
    seed = 19595
    config_systole = Config(seed,
                            False,
                            '/mnt/pre-clip-mm2-X-train.npy',
                            '/mnt/pre-clip-mm2-y-train-mm2.npy',
                            '/mnt/pre-clip-mm2-meta-train.csv',
                            0,
                            8,
                            '/data/backup/mar5/stage1-systole-clip-mm2-',
                            '/data/backup/mar5/stage1-systole-clip-mm2-train-predict.npy')

    train_hard(config_systole)
    gc.collect()

    config_diastole = Config(seed,
                             False,
                            '/mnt/pre-clip-mm2-X-train.npy',
                            '/mnt/pre-clip-mm2-y-train-mm2.npy',
                            '/mnt/pre-clip-mm2-meta-train.csv',
                            1,
                            8,
                            '/data/backup/mar5/stage1-diastole-clip-mm2-',
                            '/data/backup/mar5/stage1-diastole-clip-mm2-train-predict.npy')

    train_hard(config_diastole)


def run_weighted():
    Config = namedtuple('Config', ['seed', 'weighted', 'x', 'y', 'meta', 'col', 'nb', 'w_path', 'pred_path'])
    seed = 19595
    config_systole = Config(seed,
                            True,
                            '/mnt/pre-w-mm2--X-train.npy',
                            '/mnt/pre-w-mm2--y-train-mm2.npy',
                            '/mnt/pre-w-mm2--meta-train.csv',
                            0,
                            8,
                            '/data/backup/mar5_weighted/stage1-systole-w-mm2-',
                            '/data/backup/mar5_weighted/stage1-systole-w-mm2-train-predict.npy')

    train_hard(config_systole)
    gc.collect()

    config_diastole = Config(seed,
                             True,
                            '/mnt/pre-w-mm2--X-train.npy',
                            '/mnt/pre-w-mm2--y-train-mm2.npy',
                            '/mnt/pre-w-mm2--meta-train.csv',
                            1,
                            8,
                            '/data/backup/mar5_weighted/stage1-diastole-w-mm2-',
                            '/data/backup/mar5_weighted/stage1-diastole-w-mm2-train-predict.npy')

    train_hard(config_diastole)



@click.command()
@click.option('--save-prefix', default='/data/backup/some_run')
def main(save_prefix):
    run_weighted()


if __name__ == "__main__":
    main()
