import os
import numpy as np
import pandas as pd
import dicom
import xgboost as xgb

from sklearn.preprocessing import LabelEncoder


DATA_DIR = '/data/backup/mar5/'
PREFIX   = 'pre-w-mm2--'
SEED     = 19595


def load_train_data():
    """
    Load training data from .npy files.
    """
    X = np.load(os.path.join(DATA_DIR, PREFIX + 'X-train.npy')).astype(np.float32)
    X /= 255
    y = np.load(os.path.join(DATA_DIR, PREFIX + 'y-train-mm2.npy'))
    meta = pd.read_csv(os.path.join(DATA_DIR, PREFIX + 'meta-train.csv'), sep=',')

    return X, y, meta

def load_validation_data():
    """
    Load validation data from .npy files.
    """
    X = np.load(os.path.join(DATA_DIR, PREFIX + 'X-validate.npy')).astype(np.float32)
    X /= 255
    meta = pd.read_csv(os.path.join(DATA_DIR, PREFIX + 'meta-validate.csv'), sep=',')

    return X, meta

def split_data(X, shuffle, split_ratio=0.2):
    """
    Split data into training and testing.

    :param X: X
    :param y: y
    :param split_ratio: split ratio for train and test data
    """
    if shuffle:
        np.random.seed(SEED)
        
        if type(X) == pd.core.frame.DataFrame:
            col = X.columns
            val = X.values
            np.random.shuffle(val)
            X = pd.DataFrame(val, columns=col, dtype=float)
        else:
            np.random.shuffle(X)
    
    split = ceil(X.shape[0] * split_ratio)
    X_train = X[split:]
    X_test = X[:split]

    return X_train, X_test

def load_train_prediction(sys_file='diastole-train-mm2-normed-pred2.npy',
                          dia_file='systole-train-mm2-normed-pred2.npy', prefix='/data/backup/dry-run/'):
    sys_pred = np.load(prefix + sys_file).astype(np.float32)
    dias_pred = np.load(prefix + dia_file).astype(np.float32)
    return np.hstack((sys_pred, dias_pred))

def load_test_prediction(sys_file='systole-validate-mm2-normed-pred.npy',
                    dia_file='diastole-validate-mm2-normed-pred.npy', prefix='/data/backup/dry-run/'):
    sys_pred = np.load(prefix + sys_file).astype(np.float32)
    dias_pred = np.load(prefix + dia_file).astype(np.float32)
    return np.hstack((sys_pred, dias_pred))


le_Sex     = LabelEncoder().fit(['M', 'F'])
le_Metrics = LabelEncoder().fit(['Y', 'M', 'W'])

def df_preprocess(df, le_Sex=le_Sex, le_Metrics=le_Metrics):
    df.rename(columns = {'id':'Id', 'mm2':'Mult', 'w':'Weight', 'age':'Age', 'sex':'Sex', 'path':'Path'}, \
              inplace = True)
    df['Metrics'] = df.Age.apply(lambda x : x[-1])
    df['Age'] = df.Age.apply(lambda x : int(x[:-1]))
    
    df['Sex'] = le_Sex.transform(df.Sex)
    df['Metrics'] = le_Metrics.transform(df.Metrics)
    
    # df['Slice_Id'] = df.Path.apply(lambda x : x.split('/')[-2].split('_')[-1])
    df['Previous_Id'] = df.Id.apply(lambda x : x - 1)
    df = pd.merge(df, pd.DataFrame({'Unique_Count_Cum_Sum' : df.groupby('Id').count().Mult.cumsum()}),
                  how='left', left_on = 'Previous_Id', right_index=True).fillna(100)
    df['Slice_No'] = np.ones(df.shape[0]).cumsum() % df['Unique_Count_Cum_Sum'].values
    df.drop(['Previous_Id', 'Unique_Count_Cum_Sum'], axis=1, inplace=True)
    
    df['Adult'] = df.Age.apply(lambda x : 1 if x > 18 else 0)
    
    df.drop('Path', axis=1, inplace=True)
    
    return df.astype(float, raise_on_error=False)

def make_submission(ids, s_preds, d_preds):
    
    fi = csv.reader(open(os.path.join(DATA_DIR, 'sample_submission_validate.csv'), 'r'))
    with open(SUB_NAME, 'w') as f:
        fo = csv.writer(f, lineterminator='\n')
        fo.writerow(next(fi))
        for line in fi:
            idx = line[0]
            index, target = idx.split('_')
            index = int(index)
            out = [idx]
            if index in ids:
                if target == 'Diastole':
                    out.extend(list(d_preds[ids==index][-1]))
                else:
                    out.extend(list(s_preds[ids==index][-1]))
            else:
                print('Miss {0}'.format(idx))
            fo.writerow(out)
            
def train_xgb():
    _, y_train, meta_train = load_train_data()

    y_pred = load_train_prediction()
    meta_train['Systole_Normed_Pred'], meta_train['Diastole_Normed_Pred'] = y_pred[:, 0], y_pred[:, 1]
    del y_pred

    meta_train['Systole_Pred']  = meta_train['Systole_Normed_Pred'] * meta_train.mm2
    meta_train['Diastole_Pred'] = meta_train['Diastole_Normed_Pred'] * meta_train.mm2
    meta_train['Systole']  = y_train[:, 0] * meta_train.mm2
    meta_train['Diastole'] = y_train[:, 1] * meta_train.mm2

    meta_train = df_preprocess(meta_train)
    meta_train = pd.merge(meta_train, 
                          meta_train.groupby('Sex').mean()[['Systole', 'Diastole']], 
                          how='left', left_on='Sex', right_index=True, suffixes=('', '_Sex'))
    meta_train = pd.merge(meta_train, 
                          meta_train.groupby(['Sex', 'Adult']).mean()[['Systole', 'Diastole']],
                          how='left', left_on=['Sex', 'Adult'], right_index=True, suffixes=('', '_Sex_Adult'))

    _, meta_train = split_data(meta_train, shuffle=True)

    features = ['Age', 'Systole_Sex', 'Diastole_Sex', 'Systole_Sex_Adult', 'Diastole_Sex_Adult', 'Adult', 'Metrics',
                'Systole_Normed_Pred', 'Diastole_Normed_Pred', 'Systole_Pred', 'Diastole_Pred', 'Mult', 'Weight']
    
    X = meta_train[features].values
    y = meta_train[['Systole', 'Diastole']].values

    param = {'objective':'reg:linear', 'num_round':1000}
    num_trees = 200
    s_model = xgb.train(param, xgb.DMatrix(X, y[:, 0], feature_names=features), num_trees)
    d_model = xgb.train(param, xgb.DMatrix(X, y[:, 1], feature_names=features), num_trees)
    X = xgb.DMatrix(X, feature_names=features)

    s_sigma = np.mean(np.abs(y[:, 0] - s_model.predict(X)))
    d_sigma = np.mean(np.abs(y[:, 1] - d_model.predict(X)))
    return s_model, d_model, s_sigma, d_sigma


def predict_xgb(s_model, d_model, s_sigma, d_sigma):
    _, meta_test = load_validation_data()

    y_pred = load_test_prediction()
    meta_test['Systole_Normed_Pred']  = y_pred[:, 0]
    meta_test['Diastole_Normed_Pred'] = y_pred[:, 1]
    del y_pred
    meta_test['Systole_Pred']  = meta_test['Systole_Normed_Pred'] * meta_test.mm2
    meta_test['Diastole_Pred'] = meta_test['Diastole_Normed_Pred'] * meta_test.mm2

    meta_test = df_preprocess(meta_test)

    meta_test = pd.merge(meta_test, 
                         meta_train.groupby('Sex').mean()[['Systole_Sex', 'Diastole_Sex']], 
                         how='left', left_on='Sex', right_index=True)
    meta_test = pd.merge(meta_test, 
                         meta_train.groupby(['Sex', 'Adult']).mean()[['Systole_Sex_Adult', 'Diastole_Sex_Adult']], 
                         how='left', left_on=['Sex', 'Adult'], right_index=True)

    X = meta_test[features].values
    ids = meta_test.Id.values
    ids_short = meta_test.Id.unique().astype(int)

    X = xgb.DMatrix(X, feature_names=features)

    s_patient_preds = real_to_cdf(average_predictions(ids, s_model.predict(X)))
    d_patient_preds = real_to_cdf(average_predictions(ids, d_model.predict(X)))

    make_submission(ids_short, s_patient_preds, d_patient_preds)
    
    
if __name__ == "__main__":
    s_model, d_model, s_sigma, d_sigma = train_xgb()
    ids, s_preds, d_preds = predict_xgb()
    make_submission(ids, s_preds, d_preds)