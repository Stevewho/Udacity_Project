import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

def features_type(train):
    data = []
    for f in train.columns:
        
        # Defining the level
        if 'bin' in f or f == 'target':
            level = 'binary'
        elif 'cat' in f or f == 'id':
            level = 'nominal'
        elif train[f].dtype == float:
            level = 'interval'
        elif train[f].dtype == int:
            level = 'ordinal'
            
        # Initialize keep to True for all variables except for id
        keep = True
        if f == 'id':
            keep = False
        
        # Defining the data type 
        dtype = train[f].dtype
        
        # Creating a Dict that contains all the metadata for the variable
        f_dict = {
            'varname': f,
            'level': level,
            'keep': keep,
            'dtype': dtype
        }
        data.append(f_dict)
    meta = pd.DataFrame(data, columns=['varname', 'level', 'keep', 'dtype'])
    meta.set_index('varname', inplace=True)
    interval = meta[(meta.level == 'interval') & (meta.keep)].index
    ordinal = meta[(meta.level == 'ordinal') & (meta.keep)].index
    binary = meta[(meta.level == 'binary') & (meta.keep)].index
    nominal  = meta[(meta.level == 'nominal') & (meta.keep)].index
    return interval, ordinal, binary, nominal


def Gini(y_true, y_pred):
    # check and get number of samples
    assert y_true.shape == y_pred.shape
    n_samples = y_true.shape[0]

    # sort rows on prediction column
    # (from largest to smallest)
    arr = np.array([y_true, y_pred]).transpose()
    true_order = arr[arr[:, 0].argsort()][::-1, 0]
    pred_order = arr[arr[:, 1].argsort()][::-1, 0]

    # get Lorenz curves
    L_true = np.cumsum(true_order) * 1. / np.sum(true_order)
    L_pred = np.cumsum(pred_order) * 1. / np.sum(pred_order)
    L_ones = np.linspace(1 / n_samples, 1, n_samples)

    # get Gini coefficients (area between curves)
    G_true = np.sum(L_ones - L_true)
    G_pred = np.sum(L_ones - L_pred)

    # normalize to true Gini coefficient
    return G_pred * 1. / G_true


def feature_import(train, train_label):
    FImodel = XGBClassifier()
    FImodel.fit(train, train_label)
    sorted_idx = np.argsort(FImodel.feature_importances_)[::-1]
    #extract first 15 most importacne features
    return train.columns[sorted_idx[:15]]


def Nominal_Encode(nominal_fea, train, test ):
    for c in nominal_fea:
        le = LabelEncoder()
        x = le.fit_transform(pd.concat([train[nominal_fea], test[nominal_fea]])[c])
        train[c] = le.transform(train[c])
        test[c] = le.transform(test[c])
   
    enc = OneHotEncoder()
    enc.fit(pd.concat([train[nominal_fea], test[nominal_fea]]))
    train_nominal = enc.transform(train[nominal_fea])
    test_nominal = enc.transform(test[nominal_fea])

    return train_nominal, test_nominal

def add_noise(series, noise_level):
    return series * (1 + noise_level * np.random.randn(len(series)))


def target_encode(trn_series=None,
                  tst_series=None,
                  target=None,
                  min_samples_leaf=1,
                  smoothing=1,
                  noise_level=0):
    """
    Smoothing is computed like in the following paper by Daniele Micci-Barreca
    https://kaggle2.blob.core.windows.net/forum-message-attachments/225952/7441/high%20cardinality%20categoricals.pdf
    trn_series : training categorical feature as a pd.Series
    tst_series : test categorical feature as a pd.Series
    target : target data as a pd.Series
    min_samples_leaf (int) : minimum samples to take category average into account
    smoothing (int) : smoothing effect to balance categorical average vs prior
    """
    assert len(trn_series) == len(target)
    assert trn_series.name == tst_series.name
    temp = pd.concat([trn_series, target], axis=1)
    # Compute target mean
    averages = temp.groupby(by=trn_series.name)[target.name].agg(["mean", "count"])
    # Compute smoothing
    smoothing = 1 / (1 + np.exp(-(averages["count"] - min_samples_leaf) / smoothing))
    # Apply average function to all target data
    prior = target.mean()
    # The bigger the count the less full_avg is taken into account
    averages[target.name] = prior * (1 - smoothing) + averages["mean"] * smoothing
    averages.drop(["mean", "count"], axis=1, inplace=True)
    # Apply averages to trn and tst series
    ft_trn_series = pd.merge(
        trn_series.to_frame(trn_series.name),
        averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
        on=trn_series.name,
        how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)
    # pd.merge does not keep the index so restore it
    ft_trn_series.index = trn_series.index
    ft_tst_series = pd.merge(
        tst_series.to_frame(tst_series.name),
        averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
        on=tst_series.name,
        how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)
    # pd.merge does not keep the index so restore it
    ft_tst_series.index = tst_series.index
    return add_noise(ft_trn_series, noise_level), add_noise(ft_tst_series, noise_level)


def target_pro(train, test,train_label, feature1, feature2):

    #target_pro(train, test,interval_fea, nominal_fea)
    target_encoding_columns = list(feature1) + list(feature2)
    for f in target_encoding_columns:
        train[f + "_tef"], test[f + "_tef"] = target_encode(trn_series=train[f],
                                             tst_series=test[f],
                                             target=train_label,
                                             min_samples_leaf=200,
                                             smoothing=10,
                                             noise_level=0)
    train_tef = train[[x for x in train.columns if 'tef' in x]]
    test_tef = test[[x for x in train.columns if 'tef' in x]]
    return train_tef, test_tef
