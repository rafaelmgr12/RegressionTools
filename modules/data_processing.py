from sklearn.preprocessing import KBinsDiscretizer
from scipy.sparse import hstack, vstack
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, StratifiedKFold, ShuffleSplit


def discritizer_target(y, n_bins=100):
    '''
    Discretize the target into bins, if you want the Probability Density Functions(PDF) for each single-value of the target dataset

    Parameters: 
                y: 1D-array
                bins : The number of bins for the PDF estimation, default = 100
    '''
    y = y.reshape(-1,1)
    kbins = KBinsDiscretizer(n_bins, encode="onehot", strategy="uniform")
    kbins.fit(y.reshape(-1, 1))
    y_bins = kbins.transform(y.reshape(-1, 1))
    y_total = hstack([y_bins, y])
    y_total = y_total.toarray()
    return y_total


def tts_split(X, y, size, splits):
    '''Split the data in Train and
     test using the Shuffle split'''

    rs = ShuffleSplit(n_splits=splits, test_size=size)

    rs.get_n_splits(X)

    for train_index, test_index in rs.split(X, y):
        # print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
    return X_train, X_test, y_train, y_test


def sss_plit(X, y, nsplits):
    '''This functions it to lead with unbaleced data set, and
    make new balanced splits into train and test .
    '''
    sss = StratifiedKFold(n_splits=nsplits, random_state=None, shuffle=False)

    for train_index, test_index in sss.split(X, y):
        # print("Train:", train_index, "Test:", test_index)
        original_Xtrain, original_Xtest = X.iloc[train_index], X.iloc[test_index]
        original_ytrain, original_ytest = y.iloc[train_index], y.iloc[test_index]

    return original_Xtrain, original_Xtest, original_ytrain, original_ytest
