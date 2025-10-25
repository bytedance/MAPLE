import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold, KFold

def train_test_split_func(data, sampling, test_size, metric, random_state=132):
    """
    Split dataset into training and testing sets using random or stratified sampling.

    Parameters
    ----------
    data : DataFrame
        The full dataset to be split.
    sampling : str
        Sampling method: 'random' (default) or 'stratified'.
    test_size : float
        Proportion of the dataset to include in the test split (e.g. 0.2).
    metric : str
        Column name used for stratification when sampling='stratified'.
    random_state : int, optional
        Random seed for reproducibility, default is 132.

    Returns
    -------
    train_data : DataFrame
        The sampled training dataset.
    test_data : DataFrame
        The sampled testing dataset.
    """
    assert (sampling == 'random' or (sampling == 'stratified' and metric != None)), 'Please define the metric for stratified sampling!'
    if sampling == 'random':
        train_data, test_data = train_test_split(data,
                                                 test_size=test_size,
                                                 shuffle=True, random_state=random_state)
    else:
        train_data, test_data = train_test_split(data,
                                                 test_size=test_size,
                                                 stratify=data[metric],
                                                 shuffle=True, random_state=random_state)
    
    train_data_all = train_data.reset_index(drop=True)
    test_data_all = test_data.reset_index(drop=True)
    
    train_data = train_data_all.filter(regex='^(feature_)')
    train_meta = train_data_all.filter(regex='^(?!feature_)')
    test_data = test_data_all.filter(regex='^(feature_)')
    test_meta = test_data_all.filter(regex='^(?!feature_)')
    return (train_data_all, train_data, train_meta,
            test_data_all, test_data, test_meta)


def cv_split_func(data, meta, sampling, metric, fold, random_state=132):
    """
    Generate cross-validation splits from the dataset using random or stratified sampling.

    Parameters
    ----------
    data : DataFrame
        Input features for model training.
    meta : DataFrame
        Metadata including labels or stratification features.
    sampling : str
        Sampling method: 'random' or 'stratified'.
    metric : str
        Column name in meta for stratified sampling.
    fold : int
        Number of folds for cross-validation.
    random_state : int, optional
        Random seed for reproducibility, default is 132.

    Returns
    -------
    cv_split : generator
        Cross-validation split generator yielding (train_index, val_index).
    """
    assert (sampling == 'random' or (sampling != 'random' and metric != None)), 'Please define the metric for stratified K-Fold!'
    if sampling == 'random':
        cv_split = KFold(n_splits=fold, shuffle=True, random_state=random_state).split(data)
    else:
        cv_split = StratifiedKFold(n_splits=fold, shuffle=True, random_state=random_state).split(data, meta[metric])

    return cv_split