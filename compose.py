import os
import re
import numpy as np
import csv
import gc
from tqdm import tqdm

def build_train_table(path):
    table = {}
    for f in ['train.csv', 'validate.csv']:
        with open(os.path.join(path, f), 'r') as fin:
            fin.readline()
            for line in fin:
                s = line.replace('\n', '').split(',')
                table[int(s[0])] = (float(s[1]), float(s[2]))

    return table


def proc_sample(study_id, path='/data/samples/'):
    npy = os.path.join(path, '{}.npy'.format(study_id))
    supp = os.path.join(path, '{}.csv'.format(study_id))
    imgs = np.load(npy)

    info = []
    with open(supp, 'r') as fin:
        fin.readline()
        for i, line in enumerate(fin):
            s = line.replace('\n', '').split(',')
            (_, mm2, loc_weight,
             path, method,
             rho, age, sex) = (int(s[0]), float(s[1]),
                               float(s[2]), s[3],
                               s[4], float(s[5]),
                               s[6], s[7])
            info.append((i, mm2, loc_weight, method))

    selected = []
    meta = []
    for t in info:
        # choose near middle slices
        if t[2] > 0.7 and t[3] == 'mask':
            selected.append(t[0])
            meta.append((study_id, t[1], t[2]))

    return (imgs[selected, ...], meta)


def compose():
    path = '/data/samples/'
    train_table = build_train_table('/data')

    prefix = '/data/backup/mar8/fft-mask-w-multiscale-'

    train_ids = []
    test_ids = []

    print('List samples')

    files = os.listdir(path)
    for f in files:
        r = re.search('(\d+).npy', f)
        if r is None:
            continue
        study_id = int(r.group(1))
        if study_id > 700:
            test_ids.append(study_id)
        else:
            train_ids.append(study_id)

    print('Build train')
    X = []
    metas = []
    y = []
    for i in tqdm(train_ids):
        x, info = proc_sample(i, path)
        systole, diastole = train_table[i]
        for _, mm2, loc_weight in info:
            y.append((systole / mm2, diastole / mm2, loc_weight))
            metas.append((i, mm2, loc_weight))
        for _x in x:
            X.append(_x)

    X = np.array(X)

    print('X-train, ', X.shape)
    np.save(prefix + 'X-train.npy', X)

    y = np.array(y)
    np.save(prefix + 'y-train.npy', y)

    meta_name = prefix + 'meta-train.csv'
    header = ['id', 'mm2', 'loc_weight']
    with open(meta_name, 'w') as fout:
        w = csv.writer(fout, lineterminator='\n')
        w.writerow(header)
        w.writerows(metas)

    print('Build test')
    X = []
    metas = []
    gc.collect()

    X = []
    metas = []
    y = []
    for i in tqdm(test_ids):
        x, info = proc_sample(i, path)
        for _, mm2, loc_weight in info:
            metas.append((i, mm2, loc_weight))
        for _x in x:
            X.append(_x)

    X = np.array(X)
    print('X-test, ', X.shape)
    np.save(prefix + 'X-test.npy', X)

    meta_name = prefix + 'meta-test.csv'
    header = ['id', 'mm2', 'loc_weight']
    with open(meta_name, 'w') as fout:
        w = csv.writer(fout, lineterminator='\n')
        w.writerow(header)
        w.writerows(metas)

    print('Done')

if __name__ == "__main__":
    compose()