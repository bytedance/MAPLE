# =============================================================================
# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: GPL-3.0-or-later
# =============================================================================#
# -*- coding: utf-8 -*-
# =============================================================================
# SCRIPT  : train.py
# PROJECT : MAPLE - Cancer/Non-cancer Classifier
# PURPOSE : LightGBM classifier training with nested cross-validation and
#           Bayesian hyperparameter optimization for MAPLE classifier.
#
# AUTHOR  : Liyuan Zhao
# CREATED : 2025-07-10
# UPDATED : 2025-10-10
# =============================================================================

"""
Training module for MAPLE classifier using LightGBM.

Includes:
1. `lgb_LGBMClassifier` - Initializes a LightGBM classifier from parameter dict.
2. `nested_cv_lgm` - Performs nested cross-validation with Bayesian optimization.
3. `evaluate_on_cv` - Evaluates trained LightGBM on cross-validation splits.
"""

import logging
import re
import pandas as pd
import numpy as np
import lightgbm as lgb
from lightgbm import log_evaluation, early_stopping
from sklearn.metrics import confusion_matrix
from bayes_opt import BayesianOptimization
from model.evaluate import get_metrices, get_info
from config import train_specificity


# -----------------------------------------------------------------------------
# Function: lgb_LGBMClassifier
# -----------------------------------------------------------------------------
def lgb_LGBMClassifier(params):
    """
    Initialize and return a LightGBM LGBMClassifier using the specified parameter dictionary.

    Parameters
    ----------
    params : dict
        Dictionary containing LightGBM hyperparameters.

    Returns
    -------
    model : lgb.LGBMClassifier
        Configured LightGBM classifier instance.
    """
    model = lgb.LGBMClassifier(
        boosting_type=params["boosting_type"],
        metric=params["metric"],
        objective=params["objective"],
        seed=params["seed"],
        learning_rate=params["learning_rate"],
        num_leaves=params["num_leaves"],
        min_data_in_leaf=params["min_data_in_leaf"],
        max_depth=params["max_depth"],
        lambda_l1=params["lambda_l1"],
        lambda_l2=params["lambda_l2"],
        bagging_fraction=params["bagging_fraction"],
        bagging_freq=params["bagging_freq"],
        colsample_bytree=params["colsample_bytree"],
        verbose=params["verbose"],
        model_output=params["model_output"],
        is_unbalance=params["is_unbalance"],
    )
    return model


# -----------------------------------------------------------------------------
# Function: nested_cv_lgm
# -----------------------------------------------------------------------------
def nested_cv_lgm(
    train_data_all, train_meta, max_repeat_times, cv_split, callbacks, train_specificity
):
    """
    Perform nested cross-validation with Bayesian optimization for LightGBM hyperparameters.

    Parameters
    ----------
    train_data_all : pd.DataFrame
        Full training dataset including features and labels.
    train_meta : pd.DataFrame
        Metadata columns to filter out from features.
    max_repeat_times : int
        Number of Bayesian optimization iterations.
    cv_split : generator
        Cross-validation split generator (from cv_split_func).
    callbacks : list
        Optional LightGBM callback functions (currently unused).
    train_specificity : float
        Target specificity for model optimization.

    Returns
    -------
    best_params : dict
        Best hyperparameter set found by Bayesian optimization.
    """

    def make_cv_lgm(train_data_all, cv_split, train_meta, train_specificity):
        """
        Inner function to perform CV-based scoring for Bayesian optimization.
        """

        def cv_lgm(
            learning_rate,
            num_leaves,
            min_data_in_leaf,
            max_depth,
            lambda_l1,
            lambda_l2,
            bagging_fraction,
            bagging_freq,
            colsample_bytree,
        ):
            sns_sum = 0

            for i, (train_index, val_index) in enumerate(cv_split):
                train_sel = train_data_all.iloc[train_index, :].reset_index(drop=True)
                val_sel = train_data_all.iloc[val_index, :].reset_index(drop=True)

                x_train = train_sel.drop(
                    columns=train_meta.columns.to_list(), inplace=False
                )
                y_train = train_sel["Label"].replace({"Healthy": 0, "CRC": 1})

                x_val = val_sel.drop(
                    columns=train_meta.columns.to_list(), inplace=False
                )
                y_val = val_sel["Label"].replace({"Healthy": 0, "CRC": 1})

                params = {
                    "boosting_type": "gbdt",
                    "metric": ["binary_logloss", "auc"],
                    "objective": "binary",
                    "seed": 666,
                    "learning_rate": learning_rate,
                    "num_leaves": int(num_leaves),
                    "min_data_in_leaf": int(min_data_in_leaf),
                    "max_depth": int(max_depth),
                    "lambda_l1": lambda_l1,
                    "lambda_l2": lambda_l2,
                    "bagging_fraction": bagging_fraction,
                    "bagging_freq": int(bagging_freq),
                    "colsample_bytree": colsample_bytree,
                    "verbose": -1,
                    "model_output": "probability",
                    "is_unbalance": True,
                }

                model = lgb_LGBMClassifier(params)
                model.fit(x_train, y_train)

                # Compute sensitivity on validation
                _, val_sns = get_info(model, x_val, y_val, fold_type="valid")
                sns_sum += val_sns

            # Average sensitivity across folds
            return sns_sum / len(cv_split)

        return cv_lgm

    logging.info(f"> Repeat Bayesian optimization for {max_repeat_times} times")
    best_target = 0
    best_params = {}

    for i in range(max_repeat_times):
        cv_func = make_cv_lgm(train_data_all, cv_split, train_meta, train_specificity)
        rf_bo = BayesianOptimization(
            cv_func,
            {
                "learning_rate": (0.01, 0.3),
                "num_leaves": (5, 50),
                "min_data_in_leaf": (5, 20),
                "max_depth": (3, 50),
                "lambda_l1": (0.1, 5),
                "lambda_l2": (0.1, 5),
                "bagging_fraction": (0.1, 1),
                "bagging_freq": (1, 10),
                "colsample_bytree": (0.3, 0.8),
            },
        )
        rf_bo.maximize()
        logging.info(f"Tuning - Repeat time {i + 1}: {rf_bo.max['target']}")
        if rf_bo.max["target"] > best_target:
            best_target = rf_bo.max["target"]
            best_params = rf_bo.max["params"]

    # Ensure correct types and fixed params
    params0 = {
        "boosting_type": "gbdt",
        "metric": ["binary_logloss", "auc"],
        "objective": "binary",
        "seed": 666,
        "verbose": -1,
        "model_output": "probability",
        "is_unbalance": True,
    }
    params0.update(best_params)
    # Cast integer parameters
    params0["num_leaves"] = int(params0["num_leaves"])
    params0["min_data_in_leaf"] = int(params0["min_data_in_leaf"])
    params0["max_depth"] = int(params0["max_depth"])
    params0["bagging_freq"] = int(params0["bagging_freq"])

    return params0


# -----------------------------------------------------------------------------
# Function: evaluate_on_cv
# -----------------------------------------------------------------------------
def evaluate_on_cv(train_data_all, train_meta, params, cv_split):
    """
    Evaluate model on cross-validation splits using predefined LightGBM hyperparameters.

    Parameters
    ----------
    train_data_all : pd.DataFrame
        Full training dataset including features and labels.
    train_meta : pd.DataFrame
        Metadata used for stratification or to be excluded from features.
    params : dict
        LightGBM hyperparameters.
    cv_split : generator
        Cross-validation split generator (from cv_split_func).

    Returns
    -------
    valid_fold_res : pd.DataFrame
        Predicted probabilities and fold assignments for validation folds.
    """
    valid_fold_res = pd.DataFrame()

    for i, (train_index, val_index) in enumerate(cv_split):
        train_sel = train_data_all.iloc[train_index, :]
        val_sel = train_data_all.iloc[val_index, :]
        val_meta = train_meta.iloc[val_index, :]

        x_train = train_sel.drop(columns=train_meta.columns.to_list(), inplace=False)
        y_train = train_sel["Label"].replace({"Healthy": 0, "CRC": 1})

        x_val = val_sel.drop(columns=train_meta.columns.to_list(), inplace=False)
        y_val = val_sel["Label"].replace({"Healthy": 0, "CRC": 1})

        model = lgb_LGBMClassifier(params)
        model.fit(x_train, y_train)

        _, val_sns = get_info(model, x_val, y_val, fold_type="valid_fold")
        logging.info(f"> Sensitivity of Fold {i + 1}: {round(val_sns, 4)}")

        val_meta = val_meta.copy()
        val_meta["fold"] = f"fold{i}"
        val_meta["pred_prob"] = model.predict_proba(x_val)[:, 1]
        valid_fold_res = pd.concat([valid_fold_res, val_meta], axis=0)

    return valid_fold_res
