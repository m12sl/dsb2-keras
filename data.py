from __future__ import print_function

import os
import re
import time
import dicom
import numpy as np
from scipy.misc import imresize
from tqdm import tqdm
from multiprocessing import Pool
from skimage.restoration import denoise_tv_chambolle

import csv
import click
from collections import namedtuple, defaultdict

from fourier import process_dataset_fourier
from utils import hierarchical_load

# Meta definition should placed here because of pickling in Multiprocessing
Meta = namedtuple('Meta', ['mm2', 'loc', 'age', 'sex', 'path'])
IMG_SHAPE = (64, 64)


def load_images(files):
    def crop_resize(img):
        if img.shape[0] < img.shape[1]:
            img = img.T
         # we crop image from center
        short_edge = min(img.shape[:2])
        yy = int((img.shape[0] - short_edge) / 2)
        xx = int((img.shape[1] - short_edge) / 2)
        crop_img = img[yy: yy + short_edge, xx: xx + short_edge]
        img = crop_img
        # be careful! imresize will return UINT8 array
        img = imresize(img, IMG_SHAPE)
        return img

    images = []

    dc = dicom.read_file(files[0])
    age = getattr(dc, 'PatientAge', '000Y')
    sex = getattr(dc, 'PatientSex', 'M')
    px, py = getattr(dc, 'PixelSpacing', (1.5, 1.5))
    px, py = float(px), float(py)
    w, h = dc.pixel_array.shape
    st = int(getattr(dc, 'SliceThickness', 8))
    loc = float(getattr(dc, 'SliceLocation'))
    mm2 = (px * min([w, h]) / IMG_SHAPE[0]) ** 2

    for f in files:
        try:
            dc = dicom.read_file(f)
            image = dc.pixel_array.astype(np.float32, copy=False)
            image = image / np.max(image)
            resized = crop_resize(image)
            resized = resized.astype(np.float32, copy=False)
            resized /= np.max(resized)
            denoised = denoise_tv_chambolle(resized, weight=0.1, multichannel=False)
            images.append(denoised)
        except Exception as err:
            print('ERROR', err, f)
            continue
    images = np.array(images).astype(np.float32, copy=False)

    meta = Meta(mm2, loc, age, sex, files[0])
    return (meta, images)


def process_slice(task):
    study_id, files = task
    meta, images = load_images(files)
    return (study_id, meta, images)


def process_dataset_wslice(path, prefix, jobs=4):
    """
        Try to add weights for every slice based on distance to mid
    """
    ds = 'wtf'
    if 'train' in path:
        ds = 'train'
    if 'validate' in path:
        ds = 'validate'

    h_tasks = hierarchical_load(path)
    # we can use hierarchical information, for fourier or geometric
    flat_tasks = []
    for (study_id, slices) in h_tasks:
        for s in slices:
            flat_tasks.append((study_id, s))
    print('We have {} tasks'.format(len(flat_tasks)))
    print('Process the tasks with {} workers'.format(jobs))
    t0 = time.time()
    pool = Pool(processes=jobs)
    # process_slice should return (study_id, meta, images_bulk)
    it = pool.imap_unordered(process_slice, flat_tasks)
    work = list(tqdm(it))
    pool.close()
    pool.join()
    t1 = time.time()
    print('Done for {}s'.format(t1 - t0))
    print('-' * 50)
    print('Post-process data')

    studies = defaultdict(list)

    for (study_id, meta, images) in tqdm(work):
        studies[study_id].append((meta, images))

    X = []
    metas = []
    for study_id, payload in studies.items():
        payload.sort(key=lambda t: t[0].loc)
        locs = [t[0].loc for t in payload]
        w = np.array(locs)
        rc = 0.5 * (w[-1] + w[0])
        mc = 0.55 * (w[-1] - w[0])
        w = 1 - ((w - rc) / mc) ** 2

        a, b = zip(*payload)
        payload = zip(a, b, list(w))

        # determine the w
        for meta, images, w in payload:
            # let's use hard threshold
            # if w < 0.7:
            #     continue
            X.append(images)
            metas.append((study_id, meta.mm2, w, meta.age, meta.sex, meta.path))

    # X = []
    # metas = []
    # for (study_id, meta, images) in tqdm(work):
    #     X.append(images)
    #     metas.append((study_id, meta.mm2, meta.age, meta.sex, meta.path))

    X = np.array(X)
    X *= 255.0
    X = X.astype(np.uint8, copy=False)

    fname = '{}-X-{}.npy'.format(prefix, ds)
    np.save(fname, X)

    print('{} contains prepared array with shape {} at np.uint8'.format(fname, X.shape))
    meta_name = '{}-meta-{}.csv'.format(prefix, ds)
    header = ['id', 'mm2', 'w', 'age', 'sex', 'path']
    with open(meta_name, 'w') as fout:
        w = csv.writer(fout, lineterminator='\n')
        w.writerow(header)
        w.writerows(metas)

    print('{} contains meta with {}'.format(meta_name, str(header)))
    return (fname, meta_name, metas)


def process_dataset_simple(path, prefix, jobs=4):
    ds = 'wtf'
    if 'train' in path:
        ds = 'train'
    if 'validate' in path:
        ds = 'validate'

    h_tasks = hierarchical_load(path)
    # we can use hierarchical information, for fourier or geometric
    flat_tasks = []
    for (study_id, slices) in h_tasks:
        for s in slices:
            flat_tasks.append((study_id, s))
    print('We have {} tasks'.format(len(flat_tasks)))
    print('Process the tasks with {} workers'.format(jobs))
    t0 = time.time()
    pool = Pool(processes=jobs)
    # process_slice should return (study_id, meta, images_bulk)
    it = pool.imap_unordered(process_slice, flat_tasks)
    work = list(tqdm(it))
    pool.close()
    pool.join()
    t1 = time.time()
    print('Done for {}s'.format(t1 - t0))
    print('-' * 50)
    print('Post-process data')
    t0 = time.time()
    X = []
    metas = []
    for (study_id, meta, images) in tqdm(work):
        X.append(images)
        metas.append((study_id, meta.mm2, meta.age, meta.sex, meta.path))
    X = np.array(X)
    X *= 255.0
    X = X.astype(np.uint8, copy=False)

    fname = '{}-X-{}.npy'.format(prefix, ds)
    np.save(fname, X)

    print('{} contains prepared array with shape {} at np.uint8'.format(fname, X.shape))
    meta_name = '{}-meta-{}.csv'.format(prefix, ds)
    header = ['id', 'mm2', 'age', 'sex', 'path']
    with open(meta_name, 'w') as fout:
        w = csv.writer(fout, lineterminator='\n')
        w.writerow(header)
        w.writerows(metas)

    print('{} contains meta with {}'.format(meta_name, str(header)))
    t1 = time.time()
    print('Done for {}s'.format(t1 - t0))
    return (fname, meta_name, metas)


def write_train_npy(data_dir, save_prefix, method=None):
    """
    Loads the training data set including X and y and saves it to .npy file.
    """
    print('-' * 50)
    print('Prepare the train')

    t0 = time.time()
    path = os.path.join(data_dir, 'train')
    # fname, meta_name, metas = process_dataset_simple(path, save_prefix, jobs=4)
    if method == 'fourier':
        fname, meta_name, metas = process_dataset_fourier(path, save_prefix, jobs=8)
    else:
        fname, meta_name, metas = process_dataset_wslice(path, save_prefix, jobs=8)

    t1 = time.time()
    print('Done for {:.2f}s'.format(t1 - t0))
    print('Read train table')
    t0 = time.time()
    train_table = {}
    path = os.path.join(data_dir, 'train.csv')
    with open(path, 'r') as fin:
        for line in fin.readlines()[1:]:
            s = line.replace('\n', '').split(',')
            train_table[int(s[0])] = (float(s[1]), float(s[2]))
    print('- . ' * 10)
    print('Construct y-train')
    y = []
    for line in metas:
        study_id = line[0]
        mm2 = line[1]
        s_volume, d_volume = train_table[study_id]
        y.append( (s_volume / mm2, d_volume / mm2))

    y = np.array(y)
    y_name = '{}-y-train-{}.npy'.format(save_prefix, 'mm2')
    np.save(y_name, y)
    print('{} contains normed volumes {}'.format(y_name, y.shape))

    t1 = time.time()
    print('Done for {:.2f}s'.format(t1 - t0))


def write_validation_npy(data_dir, save_prefix, method=None):
    """
    Loads the validation data set including X and study ids and saves it to .npy file.
    """
    print('-' * 50)
    print('Prepare the validate')

    t0 = time.time()
    path = os.path.join(data_dir, 'validate')
    # fname, meta_name, metas = process_dataset_simple(path, save_prefix, jobs=4)
    if method == 'fourier':
        fname, meta_name, metas = process_dataset_fourier(path, save_prefix, jobs=8)
    else:
        fname, meta_name, metas = process_dataset_wslice(path, save_prefix, jobs=8)
    t1 = time.time()
    print('Done for {:.2f}s'.format(t1 - t0))


@click.command()
@click.option('--data-dir', default='/data/')
@click.option('--save-prefix', default='/data/pre-mm2-')
@click.option('--method', default='wslice')
def main(data_dir, save_prefix, method):
    write_train_npy(data_dir, save_prefix, method=method)
    write_validation_npy(data_dir, save_prefix, method=method)


if __name__ == "__main__":
    main()
