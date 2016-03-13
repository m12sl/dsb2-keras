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

# last skimage vs previous version
try:
    from skimage.filters import gaussian
except:
    from skimage.filters import gaussian_filter as gaussian

from scipy.misc import imresize

from utils import hierarchical_load
from skimage.restoration import denoise_tv_chambolle


HOPE_MULTIPLIER = 1.5
MAX_REGRET_ITERATIONS = 10
SCALE_FACTOR = 1.5
IMG_SHAPE = (64, 64)

Meta = namedtuple('Meta', ['corner', 'mat', 'loc', 'kx', 'ky', 'iloc', 'age', 'sex', 'path'])
Sample = namedtuple('Sample', ['id', 'img', 'mm2', 'loc_weight', 'meta', 'method', 'rho'])

NSample = namedtuple('NSample', ['id', 'imgs', 'method', 'mm2', 'meta', 'rho', 'loc2'])


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


    uniq_slices = []
    uniq_ilocs = set()
    for t in data:
        loc = t[1].iloc
        if loc in uniq_ilocs:
            continue
        uniq_slices.append(t)
        uniq_ilocs.add(loc)
    # now we process only uniq_slices

    gc.collect()
    images, metas = zip(*uniq_slices)

    h1s = [get_h1(x) for x in images]

    # CollectionViewer([x/np.max(x) for x in h1s]).show()

    # in h1s we have first harmonic
    # don't stack them, because of rare bad series

    # threashold for 5% of maximum just as in the FOURIER Guide
    m = max([np.max(h1) for h1 in h1s])
    min_threshold = 0.05 * m

    tmp = []
    for h1 in h1s:
        t = h1.copy()
        t[t < min_threshold] = 0.0
        tmp.append(t)
    h1s = tmp

    prev_center = np.zeros(3)
    center = np.zeros(3)
    rho_threshold = 200.0

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

    # ok, here we have: center, rho and some staff in metas/params/...
    # let's recompute u,v,rho_pix for every slice, not only in uniq locations

    images, metas = zip(*data)
    h1s = [get_h1(x) for x in images]

    m = max([np.max(h1) for h1 in h1s])

    methods = ['std', 'fft-c', 'fft-rho', 'h1-att', 'h1-clip']
    samples = []
    #
    for (img, meta, h1) in zip(images, metas, h1s):
        q = np.cross(meta.mat[0, :], meta.mat[1, :])
        w = center - meta.corner

        mu = np.dot(meta.mat[0, :], np.cross(w, q)) / np.dot(meta.mat[0, :], np.cross(meta.mat[1,:], q))
        la = np.dot(meta.mat[1, :], np.cross(w, q)) / np.dot(meta.mat[1, :], np.cross(meta.mat[0,:], q))

        # for debug purpose
        r_proj = la * meta.mat[0, :] + mu * meta.mat[1, :]

        offset = center - r_proj

        # coordinates to pixels
        u0, v0 = int(la / meta.kx), int(mu / meta.ky)
        rho_pixel = int(rho_threshold / meta.kx)

        # (u0, v0) is the projection of center to slice plane in pixels
        # rho_pixel is just Rho in pixels.
        # In cylindric model we will take axis vector into account, but this is spherical model.

        # here we have:
        # img -- stack of images (30, W, H) for slice
        # meta -- info for first image, location, for example
        # h1 -- fft[1] for this stack, for attend or whatever
        # u0, v0, rho_pixels for cropping
        # offset is the distance in mm between center of masses and center of projection.

        rho_pixel_max = min(u0, img.shape[1] - u0 - 1, v0, img.shape[2] - v0 - 1)

        for method in methods:
            if method == 'std':
                # this is standard preprocessing: cropp about the center
                resized = np.array([crop_resize(x) for x in img]).astype(np.float32, copy=False)
                resized /= np.max(resized)
                denoised = denoise_tv_chambolle(resized, weight=0.1, multichannel=False)
                # ['id', 'imgs', 'method', 'mm2', 'meta', 'rho', 'loc2']
                mm2 = (meta.kx * min(img.shape[1:]) / IMG_SHAPE[0]) ** 2
                # ['id', 'imgs', 'method', 'mm2', 'meta', 'rho', 'loc2'])
                samples.append(NSample(study_id, denoised, method, mm2, meta, rho_threshold, offset))

            if method == 'fft-c':
                # same as above, but centered on u0, v0

                cropped = img[:,
                              u0 - rho_pixel_max: u0 + rho_pixel_max + 1,
                              v0 - rho_pixel_max: v0 + rho_pixel_max + 1]
                resized = np.array([imresize(x, IMG_SHAPE) for x in cropped]).astype(np.float32, copy=False)
                resized /= np.max(resized)

                denoised = denoise_tv_chambolle(resized, weight=0.1, multichannel=False)
                mm2 = (2.0 * meta.kx * rho_pixel_max / IMG_SHAPE[0]) ** 2

                samples.append(NSample(study_id, denoised, method, mm2, meta, rho_threshold, offset))

            if method == 'fft-rho':
                # only in 7% of total studies rho_pixel > rho_pixel_max, so let's take rho_threshold in use
                r = min(rho_pixel, rho_pixel_max)
                cropped = img[:,
                              u0 - r: u0 + r + 1,
                              v0 - r: v0 + r + 1]

                resized = np.array([imresize(x, IMG_SHAPE) for x in cropped]).astype(np.float32, copy=False)
                resized /= np.max(resized)

                denoised = denoise_tv_chambolle(resized, weight=0.1, multichannel=False)
                mm2 = (2.0 * meta.kx * rho_pixel_max / IMG_SHAPE[0]) ** 2

                samples.append(NSample(study_id, denoised, method, mm2, meta, rho_threshold, offset))

            if method == 'h1-att':
                # let's multiply the image to h1 weight
                r = min(rho_pixel, rho_pixel_max)
                cropped = img[:,
                              u0 - r: u0 + r + 1,
                              v0 - r: v0 + r + 1]

                h1cropped = h1[u0 - r: u0 + r + 1,
                               v0 - r: v0 + r + 1]

                u, v = np.nonzero(h1cropped)

                h1attent = np.zeros_like(cropped)
                h1attent[:, u, v] = cropped[:, u, v] * h1cropped[u, v]

                resized = np.array([imresize(x, IMG_SHAPE) for x in h1attent]).astype(np.float32, copy=False)
                resized /= np.max(resized)
                mm2 = (2.0 * meta.kx * rho_pixel_max / IMG_SHAPE[0]) ** 2

                denoised = denoise_tv_chambolle(resized, weight=0.1, multichannel=False)
                samples.append(NSample(study_id, resized, method, mm2, meta, rho_threshold, offset))
                samples.append(NSample(study_id, denoised, method + '-tv', mm2, meta, rho_threshold, offset))

            if method == 'h1-clip':
                # let's multiply the image to h1 weight
                r = min(rho_pixel, rho_pixel_max)
                cropped = img[:,
                              u0 - r: u0 + r + 1,
                              v0 - r: v0 + r + 1]

                h1cropped = h1[u0 - r: u0 + r + 1,
                               v0 - r: v0 + r + 1]

                h1cropped[h1cropped < 0.05 * m] = 0
                u, v = np.nonzero(h1cropped)

                h1attent = np.zeros_like(cropped)
                h1attent[:, u, v] = cropped[:, u, v]

                resized = np.array([imresize(x, IMG_SHAPE) for x in h1attent]).astype(np.float32, copy=False)
                resized /= np.max(resized)
                mm2 = (2.0 * meta.kx * rho_pixel_max / IMG_SHAPE[0]) ** 2

                denoised = denoise_tv_chambolle(resized, weight=0.1, multichannel=False)
                samples.append(NSample(study_id, resized, method, mm2, meta, rho_threshold, offset))
                samples.append(NSample(study_id, denoised, method + '-tv', mm2, meta, rho_threshold, offset))

    return samples


def middleware(task):
    path = '/data/nsamples/'
    study_id, slices = task
    samples = process_the_study(task)

    fname = path + '{}.npy'.format(study_id)
    sname = path + '{}.csv'.format(study_id)

    if os.path.exists(fname) and os.path.exists(sname):
        return (fname, sname)

    X = []
    supp = []
    for study_id, img, method, mm2, meta, rho, offset in samples:
        X.append(img)
        # study_id, denoised, method + '-tv', mm2, meta, rho_threshold, offset)
        supp.append((study_id, method, mm2, meta.path, rho, offset))

    X = np.array(X)
    X *= 255.0 / np.max(X)
    X = X.astype(np.uint8, copy=False)

    np.save(fname, X)

    header = ['id', 'method', 'mm2', 'path', 'rho', 'offset']
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


def testrun():
    h_tasks = hierarchical_load('../train/')
    print(len(h_tasks))

    nsamples = process_the_study(h_tasks[10])
    print('We have {} samples'.format(len(nsamples)))

    to_show = []
    for t in nsamples:
        to_show.append(t.imgs[0, :, :])
        to_show.append(t.imgs[1, :, :])
        to_show.append(t.imgs[10, :, :])
        to_show.append(t.imgs[20, :, :])

    CollectionViewer(to_show).show()


if __name__ == "__main__":
    process_all(30)
    # testrun()