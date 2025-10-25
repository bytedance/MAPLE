# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: GPL-3.0-or-later
# =============================================================================
# -*- coding: utf-8 -*-
# =============================================================================
# SCRIPT  : split.py
# PROJECT : MAPLE - Cancer/Non-cancer Classifier
# PURPOSE : Provide training/test splitting and cross-validation split functions
#           supporting both random and stratified sampling strategies.
#
# AUTHOR  : Liyuan Zhao
# CREATED : 2025-07-10
# UPDATED : 2025-10-10
# =============================================================================

"""
Data splitting utilities for the MAPLE classifier.

This module includes:
1. `train_test_split_func` - split dataset into training and test sets
   with optional stratification based on a target metric.
2. `cv_split_func` - generate cross-validation fold indices using random
   or stratified K-Folds.

Both functions reset indices and separate feature columns (starting with
'feature_') from metadata columns.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold


# -----------------------------------------------------------------------------
# Function: train_test_split_func
# -----------------------------------------------------------------------------
def train_test_split_func(data, sampling, test_size, metric, random_state=132):
    """
    Split dataset into training and testing sets using random or stratified sampling.

    Parameters
    ----------
    data : pd.DataFrame
        The full dataset to be split.
    sampling : str
        Sampling method: 'random' or 'stratified'.
    test_size : float
        Fraction of dataset to include in test set (e.g. 0.2).
    metric : str
        Column name for stratification (required if sampling='stratified').
    random_state : int, optional
        Random seed for reproducibility (default: 132).

    Returns
    -------
    train_data_all, train_data, train_meta, test_data_all, test_data, test_meta : tuple of pd.DataFrame
        - train_data_all / test_data_all: full datasets with features + metadata
        - train_data / test_data: feature-only datasets
        - train_meta / test_meta: metadata-only datasets
    """
    # Validation check
    assert sampling == "random" or (
        sampling == "stratified" and metric is not None
    ), "Please define the metric for stratified sampling!"

    # Split dataset
    if sampling == "random":
        train_data, test_data = train_test_split(
            data, test_size=test_size, shuffle=True, random_state=random_state
        )
    else:
        train_data, test_data = train_test_split(
            data,
            test_size=test_size,
            stratify=data[metric],
            shuffle=True,
            random_state=random_state,
        )

    # Reset indices
    train_data_all = train_data.reset_index(drop=True)
    test_data_all = test_data.reset_index(drop=True)

    # Separate features and metadata
    train_data = train_data_all.filter(regex="^(feature_)")
    train_meta = train_data_all.filter(regex="^(?!feature_)")
    test_data = test_data_all.filter(regex="^(feature_)")
    test_meta = test_data_all.filter(regex="^(?!feature_)")

    return train_data_all, train_data, train_meta, test_data_all, test_data, test_meta


# -----------------------------------------------------------------------------
# Function: cv_split_func
# -----------------------------------------------------------------------------
def cv_split_func(data, meta, sampling, metric, fold, random_state=132):
    """
    Generate cross-validation splits from the dataset using random or stratified sampling.

    Parameters
    ----------
    data : pd.DataFrame
        Input feature dataset.
    meta : pd.DataFrame
        Metadata including labels or stratification feature.
    sampling : str
        Sampling method: 'random' or 'stratified'.
    metric : str
        Column in `meta` for stratified sampling.
    fold : int
        Number of folds for cross-validation.
    random_state : int, optional
        Random seed for reproducibility (default: 132).

    Returns
    -------
    cv_split : generator
        Generator yielding (train_index, val_index) for each fold.
    """
    # Validation check
    assert sampling == "random" or (
        sampling != "random" and metric is not None
    ), "Please define the metric for stratified K-Fold!"

    if sampling == "random":
        cv_split = KFold(n_splits=fold, shuffle=True, random_state=random_state).split(
            data
        )
    else:
        cv_split = StratifiedKFold(
            n_splits=fold, shuffle=True, random_state=random_state
        ).split(data, meta[metric])

    return cv_split
