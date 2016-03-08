import numpy as np
from scipy.fftpack import fftn, ifftn
from collections import namedtuple
import time
from tqdm import tqdm
from multiprocessing import Pool
import gc

import os
import matplotlib.pyplot as plt

from skimage.viewer import CollectionViewer
import csv

import dicom
from skimage.draw import circle

# from skimage.filters import gaussian_filter
from skimage.filters import gaussian


from scipy.misc import imresize

from utils import hierarchical_load


HOPE_MULTIPLIER = 1.5
MAX_REGRET_ITERATIONS = 10
SCALE_FACTOR = 1.5

Meta = namedtuple('Meta', ['corner', 'mat', 'loc', 'kx', 'ky', 'iloc', 'age', 'sex', 'path'])
Sample = namedtuple('Sample', ['id', 'img', 'mm2', 'loc_weight', 'meta', 'method', 'rho'])


def centroid(img):
        points = np.nonzero(img)
        w = img[points]
        x, y = np.average(np.transpose(points), axis=0, weights=img[points])
        return x, y, np.sum(img[points])

def get_h1(imgs):
    ff = fftn(imgs)
    h1 = np.absolute(ifftn(ff[1, :, :]))
    scale = np.max(h1)
    # h1 = scale * gaussian_filter(h1 / scale, 5)
    h1 = scale * gaussian(h1 / scale, 5)
    return h1

def rho_distribution(arr, u0, v0):
    u, v = np.nonzero(arr)
    r = np.sqrt((u - u0) ** 2 + (v - v0) ** 2)
    pr = arr[u, v] * r
    return zip(pr, r)

def drop_out(arr, u0, v0, th):
    u, v = np.nonzero(arr)
    r = np.sqrt((u - u0) ** 2 + (v - v0) ** 2)
    idx = (r > th)
    newarr = arr.copy()
    newarr[u[idx], v[idx]] = 0.0
    return newarr
#
# def drop_out3(arr, u0, v0, th):
#     u, v = circle(int(u0), int(v0), int(th), arr.shape[1:])
#
#     newarr = np.zeros_like(arr)
#     newarr[:, u, v] = arr[:, u, v]
#
#     # newarr = []
#     # for x in arr:
#     #     y = np.zeros_like(x)
#     #     y[u, v] = x[u, v]
#     #     newarr.append(y)
#
#
#     return newarr


def load_slice(files):
    images = []

    dc = dicom.read_file(files[0])
    # print(dc.ImageOrientationPatient)
    # print(map(float, dc.ImageOrientationPatient))
    mat = np.array(list(map(float, dc.ImageOrientationPatient))).reshape(2, 3)
    corner = np.array(list(map(float, dc.ImagePositionPatient)))
    loc = float(dc.SliceLocation)
    kx, ky = list(map(float, dc.PixelSpacing))
    sex = dc.PatientSex
    age = dc.PatientAge

    for f in files:
        dc = dicom.read_file(f)
        img = dc.pixel_array
        images.append(img)

    images = np.array(images).astype(np.float32, copy=False)
    images /= np.max(images)

    return images, Meta(corner, mat, loc, kx, ky, int(loc), age, sex, files[0])


def process_the_study(task):
    # print study_id
    study_id, slices = task
    data = []
    for s in slices:
        data.append(load_slice(s))
    # here in data we have (images, meta) tuples

    data.sort(key=lambda t: t[1].iloc)

    # just kludge for working only with uniq slices
    unsliced = []
    tmp = set()
    for t in data:
        loc = t[1].iloc
        if loc in tmp:
            continue
        unsliced.append(t)
        tmp.add(loc)

    # todo: take the latest with the same iloc
    # todo: process dropped slices too

    data = unsliced

    gc.collect()

    locs = [t[1].iloc for t in data]

    if len(set(locs)) != len(locs):
        print("Non unique locations in {}".format(study_id))

    images, metas = zip(*data)

    h1s = [get_h1(x) for x in images]

    # CollectionViewer([x/np.max(x) for x in h1s]).show()

    # in h1s we have first harmonic
    # don't stack them, because of rare bad series

    # threashold for 5% of maximum
    m = max([np.max(h1) for h1 in h1s])
    min_threshold = 0.05 * m

    tmp = []
    for h1 in h1s:
        t = h1.copy()
        t[t < min_threshold] = 0.0
        tmp.append(t)
    h1s = tmp

    prev_center = np.zeros(3)

    counter = 0
    condition = True

    gc.collect()
    while condition:
        R = []
        W = []
        for (h1, meta) in zip(h1s, metas):
            u, v, w = centroid(h1)
            rc = meta.corner + meta.kx * u * meta.mat[0, :] + meta.ky * v * meta.mat[1, :]
            W.append(w)
            R.append(rc)
        center = np.average(R, axis=0, weights=W)
        dist = np.sqrt(np.sum((prev_center - center) ** 2))
        # print('center moved to {} mm'.format(dist))
        condition = (dist > 1.0)
        prev_center = center

        # get distance pixel-distance-distribution
        rho_dist = []
        coords = []
        for (h1, meta) in zip(h1s, metas):
            # compute the projection of center to the slice plane

            q = np.cross(meta.mat[0, :], meta.mat[1, :])
            w = center - meta.corner

            mu = np.dot(meta.mat[0, :], np.cross(w, q)) / np.dot(meta.mat[0, :], np.cross(meta.mat[1,:], q))
            la = np.dot(meta.mat[1, :], np.cross(w, q)) / np.dot(meta.mat[1, :], np.cross(meta.mat[0,:], q))

            # for debug purpose
            r_proj = la * meta.mat[0, :] + mu * meta.mat[1, :]

            # coordinates to pixels
            u0, v0 = la / meta.kx, mu / meta.ky

            # small optimization
            coords.append((u0, v0, meta.kx))

            wrho_vs_rho = rho_distribution(h1, u0, v0)

            # rescale distribution to real mm
            rho_dist.extend([(meta.kx * t[0], meta.kx * t[1]) for t in wrho_vs_rho])

        rho_dist.sort(key=lambda t: t[1])
        RD = np.array(rho_dist)
        d, r = RD[:,0], RD[:, 1]
        d = np.cumsum(d)
        # plt.plot(r, d / r)
        # plt.show()

        rho_threshold = r[np.argmax(d / r)]
        # print('rho-max: {} in real mm'.format(rho_threshold))

        # now we can drop some pixels in h1s

        new_h1s = []
        params = []
        for (h1, coord) in zip(h1s, coords):
            u0, v0, kx = coord
            threshold = HOPE_MULTIPLIER * rho_threshold / kx
            new_h1 = drop_out(h1, u0, v0, threshold)

            params.append((u0, v0, threshold))
            new_h1s.append(new_h1)

        h1s = new_h1s

        counter += 1
        if counter > MAX_REGRET_ITERATIONS:
            break

    gc.collect()

    clean_images = []

    samples = []

    # let's computer params for quadric weights
    w = np.array([meta.loc for meta in metas])
    rc = 0.5 * (w[-1] + w[0])
    mc = 0.55 * (w[-1] - w[0])

    for image, h1, meta, param in zip(images, h1s, metas, params):
        loc_weight = 1.0 - ((meta.loc - rc) / mc) ** 2
        u0, v0, th = param
        r_min = 32
        r_max = min(u0, image.shape[1] - u0 - 1, v0, image.shape[2] - v0 - 1)

        # just simple circle mask
        # circle-masked

        # clean_images.append(circle_masked[0, :, :] / np.max(circle_masked[0, :, :]))

        # h1-masked
        # mat = np.zeros_like(image)
        # u, v = np.nonzero(h1)
        # mat[:, u, v] = image[:, u, v]

        # # h1-attented
        # mat = np.zeros_like(image)
        # mat[:, u, v] = image[:, u, v] * h1[u, v]

        #
        for method in ['mask', 'h1_mask', 'h1_attend']:
            mat = np.zeros_like(image)

            if method == 'mask':
                u, v = circle(int(u0), int(v0), int(th), image.shape[1:])
                mat[:, u, v] = image[:, u, v]

            if method == 'h1_mask':
                u, v = np.nonzero(h1)
                mat[:, u, v] = image[:, u, v]

            if method == 'h1_attend':
                u, v = np.nonzero(h1)
                mat[:, u, v] = image[:, u, v] * h1[u, v]


            r = min(r_max, th)
            while r >= r_min:
                x = mat[:, u0 - r: u0 + r + 1, v0 - r: v0 + r + 1]

                # don't forget about common sense
                mm2 = (meta.kx * r / 32) ** 2
                # clean_images.append(x[0, :, :] / np.max(x))

                # be aware, imresize will scale image to 255.
                img_cnn = np.array([imresize(y, (64, 64)) for y in x]).astype(np.float32, copy=False)
                img_cnn /= min(255.0, np.max(img_cnn))
                # this is sample
                # Sample = namedtuple('Sample', ['id', 'img', 'mm2', 'loc_weight', 'meta', 'method', 'rho'])
                samples.append(Sample(study_id, img_cnn, mm2, loc_weight, meta, method, r))

                r /= SCALE_FACTOR

    # CollectionViewer(clean_images).show()
    return samples


def middleware(task):
    path = '/data/samples/'
    study_id, slices = task
    samples = process_the_study(task)

    fname = path + '{}.npy'.format(study_id)
    sname = path + '{}.csv'.format(study_id)

    if os.path.exists(fname) and os.path.exists(sname):
        return (fname, sname)

    X = []
    supp = []
    for study_id, img, mm2, loc, meta, method, r in samples:
        X.append(img)
        supp.append((study_id, mm2, loc, meta.path, method, r, meta.age, meta.sex))

    X = np.array(X)
    X *= 255.0 / np.max(X)
    X = X.astype(np.uint8, copy=False)

    np.save(fname, X)

    header = ['id', 'mm2', 'loc_weight', 'path', 'method', 'rho', 'age', 'sex']
    with open(sname, 'w') as fout:
        w = csv.writer(fout, lineterminator='\n')
        w.writerow(header)
        w.writerows(supp)

    return (fname, sname)


def process_all(jobs=8):
    t0 = time.time()
    tasks = []
    for path in ['/data/train', '/data/validate', '/data/test']:
        h_tasks = hierarchical_load(path)
        tasks.extend(h_tasks)
    t1 = time.time()
    print('{} tasks loaded for {}s'.format(len(tasks), t1 - t0))

    t0 = time.time()

    pool = Pool(processes=jobs)
    it = pool.imap_unordered(middleware, tasks)
    work = list(tqdm(it))

    pool.close()
    pool.join()
    t1 = time.time()

    print('Done for {}s'.format(t1 - t0))


def process_dataset_fourier(path, prefix, jobs=4):
    ds = 'wtf'
    if 'train' in path:
        ds = 'train'
    if 'validate' in path:
        ds = 'validate'

    h_tasks = hierarchical_load(path)
    print(len(h_tasks))
    t0 = time.time()

    # samples = process_the_study(h_tasks[0])

    pool = Pool(processes=jobs)
    # process_slice should return (study_id, meta, images_bulk)
    it = pool.imap_unordered(process_the_study, h_tasks)
    work = list(tqdm(it))
    pool.close()
    pool.join()

    t1 = time.time()
    print("Done for {}s".format(t1 - t0))

    done = []
    for w in work:
        done.extend(w)

    work = done

    X = []
    metas = []
    for sample in work:
        study_id, images, mm2, loc_weight, meta = sample
        X.append(images)
        metas.append((study_id, mm2, loc_weight, meta.age, meta.sex, meta.path))

    # Memory Error will kill all work \=
    # So, let's try the kludge
    gc.collect()

    meta_name = '{}-meta-{}.csv'.format(prefix, ds)
    header = ['id', 'mm2', 'loc_weight', 'age', 'sex', 'path']
    with open(meta_name, 'w') as fout:
        w = csv.writer(fout, lineterminator='\n')
        w.writerow(header)
        w.writerows(metas)

    print('{} contains meta with {}'.format(meta_name, str(header)))
    del metas
    gc.collect()

    # We can try to make X by sequence of pop and concats

    X = np.array(X)
    X *= 255.0
    X = X.astype(np.uint8, copy=False)
    fname = '{}-X-{}.npy'.format(prefix, ds)

    np.save(fname, X)
    print('{} data saved to {} with shape {} at np.uint8'.format(ds, fname, X.shape))


    return (fname, meta_name, metas)


if __name__ == "__main__":
    process_all(8)
