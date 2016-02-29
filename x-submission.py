from __future__ import print_function

import csv
import numpy as np

from model import get_model
from utils import real_to_cdf, preprocess

import os
DATA_DIR = '/mnt/dsb2-keras/dry-run/'


def load_validation_data():
    """
    Load validation data from .npy files.
    """
    X = np.load(os.path.join(DATA_DIR, 'pre-X-validate.npy'))
    m = np.load(os.path.join(DATA_DIR, 'pre-m-validate.npy'))
    ids = m[:, 0].astype(int)

    # let's take mm2
    mult = m[:, 1]

    # let's take mm3
    # mult = m[:, 2]

    X = X.astype(np.float32)
    X /= 255

    return X, ids, mult


def accumulate_study_results(ids, prob):
    """
    Accumulate results per study (because one study has many SAX slices),
    so the averaged CDF for all slices is returned.
    """
    sum_result = {}
    cnt_result = {}
    size = prob.shape[0]
    for i in range(size):
        study_id = ids[i]
        idx = int(study_id)
        if idx not in cnt_result:
            cnt_result[idx] = 0.
            sum_result[idx] = np.zeros((1, prob.shape[1]), dtype=np.float32)
        cnt_result[idx] += 1
        sum_result[idx] += prob[i, :]
    for i in cnt_result.keys():
        sum_result[i][:] /= cnt_result[i]
    return sum_result


def build_submission(config):
    model_systole = get_model()
    model_diastole = get_model()

    print('Loading models weights...')

    model_systole.load_weights(config.systole_weights)
    model_diastole.load_weights(config.diastole_weights)


    # load val losses to use as sigmas for CDF
    with open(config.val_loss_systole, 'r') as f:
        val_loss_systole = float(f.readline())

    with open(config.val_loss_diastole, 'r') as f:
        val_loss_diastole = float(f.readline())

    print('Loading validation data...')
    X, ids, mult = load_validation_data()

    batch_size = 32
    print('Predicting on validation data...')


    pred_normed_systole = model_systole.predict(X, batch_size=batch_size, verbose=1)
    pred_normed_diastole = model_diastole.predict(X, batch_size=batch_size, verbose=1)

    print('Normed_systole:', pred_normed_systole.shape)
    print('Normed_diastole:', pred_normed_diastole.shape)

    print('mult:', mult.shape)

    pred_systole = pred_normed_systole[:,0] * mult
    pred_diastole = pred_normed_diastole[:,0] * mult

    print('systole:', pred_systole.shape)
    print('diastole:', pred_diastole.shape)


    # real predictions to CDF
    cdf_pred_systole = real_to_cdf(pred_systole, val_loss_systole)
    cdf_pred_diastole = real_to_cdf(pred_diastole, val_loss_diastole)

    print('Accumulating results...')
    sub_systole = accumulate_study_results(ids, cdf_pred_systole)
    sub_diastole = accumulate_study_results(ids, cdf_pred_diastole)

    # write to submission file
    print('Writing submission to file...')
    fi = csv.reader(open('/data/sample_submission_validate.csv'))
    f = open(config.submission, 'w')
    fo = csv.writer(f, lineterminator='\n')
    fo.writerow(next(fi))
    for line in fi:
        idx = line[0]
        key, target = idx.split('_')
        key = int(key)
        out = [idx]
        if key in sub_systole:
            if target == 'Diastole':
                out.extend(list(sub_diastole[key][0]))
            else:
                out.extend(list(sub_systole[key][0]))
        else:
            print('Miss {0}'.format(idx))
        fo.writerow(out)
    f.close()

    print('Done.')

class Config(object):
    pass

if __name__ == "__main__":
    prefix = '/mnt/dsb2-keras/dry-run/'

    config = Config()
    config.systole_weights = prefix + '19595-2-mm2-weights_systole_best.hdf5'
    config.diastole_weights = prefix + '19595-3-mm2-weights_diastole_best.hdf5'
    config.val_loss_systole = prefix + '19595-2-mm2-val_loss.txt'
    config.val_loss_diastole = prefix + '19595-3-mm2-val_loss.txt'
    config.submission = prefix + 'mm2-submission.csv'

    # config.systole_weights = prefix + '19595-4-mmx-weights_systole_best.hdf5'
    # config.diastole_weights = prefix + '19595-5-mmx-weights_diastole_best.hdf5'
    # config.val_loss_systole = prefix + '19595-4-mmx-val_loss.txt'
    # config.val_loss_diastole = prefix + '19595-5-mmx-val_loss.txt'
    # config.submission = prefix + 'mm3-submission.csv'

    build_submission(config)




"""
19595-0-mmx-weights_systole_best.hdf5
19595-0-mmx-weights_systole.hdf5
19595-2-0.05lr-mmx-val_loss.txt
19595-2-0.05lr-mmx-weights_systole_best.hdf5
19595-2-0.05lr-mmx-weights_systole.hdf5
19595-2-mm2-val_loss.txt
19595-2-mm2-weights_systole_best.hdf5
19595-2-mm2-weights_systole.hdf5
19595-3-mm2-val_loss.txt
19595-3-mm2-weights_diastole_best.hdf5
19595-3-mm2-weights_diastole.hdf5
19595-4-mmx-val_loss.txt
19595-4-mmx-weights_systole_best.hdf5
19595-4-mmx-weights_systole.hdf5
19595-5-mm3-weights_diastole_best.hdf5
19595-5-mm3-weights_diastole.hdf5
19595-5-mmx-val_loss.txt
19595-5-mmx-weights_diastole_best.hdf5
19595-5-mmx-weights_diastole.hdf5
19595-pre-mm2-weights_systole_best.hdf5
19595-pre-mm2-weights_systole.hdf5
19595-pre-weights_systole_best.hdf5
19595-pre-weights_systole.hdf5
pre-m-validate.npy
pre-X-train.npy
pre-X-validate.npy
pre-y-train.npy
"""