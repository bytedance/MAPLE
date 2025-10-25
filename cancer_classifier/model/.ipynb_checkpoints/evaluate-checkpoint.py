import os
import argparse
import logging
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from bayes_opt import BayesianOptimization
from config import train_specificity

def get_metrices(y_true, y_pred, metric):
    """
    Compute performance metrics: sensitivity, specificity, or accuracy.

    Parameters
    ----------
    y_true : array-like
        True class labels.
    y_pred : array-like
        Predicted class labels.
    metric : str
        Metric to compute: 'sns' (sensitivity), 'spe' (specificity), or 'accuracy'.

    Returns
    -------
    score : float
        Computed value for the selected metric.
    """
    assert metric in ['sns', 'spe', 'accuracy']
    confusion = confusion_matrix(y_true, 
                                 y_pred)
    TP = confusion[1, 1]
    TN = confusion[0, 0]
    FP = confusion[0, 1]
    FN = confusion[1, 0]
    if metric == 'sns':
        return TP / float(TP + FN)
    elif metric == 'spe':
        return TN / float(TN + FP)
    elif metric == 'accuracy':
        return (TP + TN) / float(TP + TN + FP + FN)
    

def get_info(model, x_val, y_val, fold_type):
    """
    Compute classification threshold and sensitivity on validation set.

    Parameters
    ----------
    model : sklearn-like classifier
        Trained model with predict_proba method.
    x_val : DataFrame
        Validation features.
    y_val : array-like
        Validation labels.
    fold_type : str
        Type of fold: 'train' or 'valid' (used for logging/debugging).

    Returns
    -------
    result : list
        List containing the decision threshold and corresponding sensitivity.
    """
    pred_val = model.predict_proba(x_val)[:, 1]
    control_pred = pred_val[y_val==0]
    try:
        cut_off = min(control_pred[control_pred>np.quantile(control_pred, train_specificity)])
    except:
        cut_off = np.quantile(control_pred, train_specificity)

    y_pred = np.array([1 if y > cut_off else 0 for y in pred_val])
    sns = get_metrices(y_val, y_pred, 'sns')
    return([cut_off, sns])


def cal_sensitivity_at_specificity(y_true, 
                                   y_pred_prob, 
                                   specificity_threshold, 
                                   intervals):
    """
    Returns the maximum of sensitivity that satisfies the constraint of specificity >= threshold.

    Sensitivity: the ability of a test to correctly identify patients with a disease. TP / (TP + FN)
    Specificity: the ability of a test to correctly identify people without the disease. TN / (TN + FP)

    Parameters:
    ----------
        y_true: Pandas DataFrame columns, str
            true label in given set
        y_pred_prob: Pandas DataFrame columns, float
            predicted probability in given set
        specificity_threshold: float
            A scalar value in range `[0, 1]`.
        intervals: Int
            Number of intervals for searching the maximum sensitivity. Defaults: 1000.

    Returns:
    -------
        new threshold: float
            Maximal dependent value.
        new specificity: float
            Minimum specificity beyond the given specificity_threshold with given step:
        new sensitivity: float
            Sensitivity at the specificity
    """
    pred_specificity = 0
    step = (y_pred_prob.max() - y_pred_prob.min()) / intervals
    k = y_pred_prob.min() + step
    while pred_specificity < specificity_threshold:
        if k < y_pred_prob.max():
            y_pred = [1 if y >= k else 0 for y in y_pred_prob]
            tn, fp, fn, tp = confusion_matrix(
                y_true.tolist(), y_pred, labels=[0, 1]
            ).ravel()
            pred_specificity = tn / (tn + fp)
            pred_sensitivity = tp / (tp + fn)
        else:
            k = y_pred_prob.max()
            y_pred = [1 if y >= k else 0 for y in y_pred_prob]
            tn, fp, fn, tp = confusion_matrix(
                y_true.tolist(), y_pred, labels=[0, 1]
            ).ravel()
            pred_specificity = tn / (tn + fp)
            pred_sensitivity = tp / (tp + fn)
            break
        k += step
    return (pred_specificity, pred_sensitivity, k)