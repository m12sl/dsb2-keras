from __future__ import print_function

import numpy as np
from scipy.stats import norm
from skimage.restoration import denoise_tv_chambolle
from scipy import ndimage
from keras.utils.generic_utils import Progbar
import os
import re


def crps(true, pred):
    """
    Calculation of CRPS.

    :param true: true values (labels)
    :param pred: predicted values
    """
    return np.sum(np.square(true - pred)) / true.size

def real_to_cdf_var(y, vsigma):
    cdf = np.zeros((y.shape[0], 600))
    for i in range(y.shape[0]):
        cdf[i] = norm.cdf(np.linspace(0, 599, 600), y[i], vsigma[i])
    return cdf


def real_to_cdf(y, sigma=1e-10):
    """
    Utility function for creating CDF from real number and sigma (uncertainty measure).

    :param y: array of real values
    :param sigma: uncertainty measure. The higher sigma, the more imprecise the prediction is, and vice versa.
    Default value for sigma is 1e-10 to produce step function if needed.
    """
    cdf = np.zeros((y.shape[0], 600))
    for i in range(y.shape[0]):
        cdf[i] = norm.cdf(np.linspace(0, 599, 600), y[i], sigma)
    return cdf


def preprocess(X):
    """
    Pre-process images that are fed to neural network.

    :param X: X
    """
    progbar = Progbar(X.shape[0])  # progress bar for pre-processing status tracking

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            X[i, j] = denoise_tv_chambolle(X[i, j], weight=0.1, multichannel=False)
        progbar.add(1)
    return X


def rotation_augmentation(X, angle_range):
    progbar = Progbar(X.shape[0])  # progress bar for augmentation status tracking

    X_rot = np.copy(X)
    for i in range(len(X)):
        angle = np.random.randint(-angle_range, angle_range)
        for j in range(X.shape[1]):
            X_rot[i, j] = ndimage.rotate(X[i, j], angle, reshape=False, order=1)
        progbar.add(1)
    return X_rot


def shift_augmentation(X, h_range, w_range):
    progbar = Progbar(X.shape[0])  # progress bar for augmentation status tracking

    X_shift = np.copy(X)
    size = X.shape[2:]
    for i in range(len(X)):
        h_random = np.random.rand() * h_range * 2. - h_range
        w_random = np.random.rand() * w_range * 2. - w_range
        h_shift = int(h_random * size[0])
        w_shift = int(w_random * size[1])
        for j in range(X.shape[1]):
            X_shift[i, j] = ndimage.shift(X[i, j], (h_shift, w_shift), order=0)
        progbar.add(1)
    return X_shift



def read_sax_folder(path):
    files = []
    for x in os.listdir(path):
        r = re.search('(\d+)-(\d+)\.dcm', x)
        if r is None:
            continue
        m = int(r.group(2))
        files.append((m, x))

    files = [os.path.join(path, t[1]) for t in sorted(files)]
    return files


def hierarchical_load(path):
    all_studies = []
    for x in os.listdir(path):
        r = re.match('\d+', x)
        if r is None:
            continue
        else:
            all_studies.append((int(r.group()), x))

    all_studies.sort()
    studies = []

    for (study_id, study_dir) in all_studies:
        p = os.path.join(path, study_dir, 'study')
        study_saxes_paths = []
        for x in os.listdir(p):
            r = re.match('sax_(\d+)', x)
            if r is None:
                continue
            m = int(r.group(1))
            study_saxes_paths.append((m, x))

        study_saxes_paths = [os.path.join(p, t[1]) for t in sorted(study_saxes_paths)]
        study_slices_paths = []

        for sax_path in study_saxes_paths:
            sax_files = read_sax_folder(sax_path)
            if len(sax_files) == 30:
                study_slices_paths.append(sax_files)
            if len(sax_files) < 30:
                study_slices_paths.append(sax_files + sax_files[:30 - len(sax_files)])
            if len(sax_files) > 30:
                study_slices_paths.extend(zip(*[iter(sax_files)] * 30))
        # we want not saxes, but slices!
        studies.append((study_id, study_slices_paths))
    return studies