# =============================================================================
# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: GPL-3.0-or-later
# =============================================================================
# -*- coding: utf-8 -*-
# =============================================================================
# SCRIPT  : evaluate.py
# PROJECT : MAPLE - Cancer/Non-cancer Classifier
# PURPOSE : Compute sensitivity, specificity, and thresholds for LightGBM
#           classifier evaluation.
#
# AUTHOR  : Liyuan Zhao
# CREATED : 2025-07-10
# UPDATED : 2025-10-10
# =============================================================================


"""
Evaluation utilities for MAPLE classifier.

Includes:
1. `get_metrices` - Compute sensitivity, specificity, or accuracy.
2. `get_info` - Compute threshold and sensitivity for a given validation set.
3. `cal_sensitivity_at_specificity` - Find threshold to satisfy a specificity constraint.
"""


# -----------------------------------------------------------------------------
# Function: get_metrices
# -----------------------------------------------------------------------------
def get_metrices(y_true, y_pred, metric):
    """
    Compute classification performance metrics: sensitivity, specificity, or accuracy.

    Parameters
    ----------
    y_true : array-like
        True class labels (0 or 1).
    y_pred : array-like
        Predicted class labels (0 or 1).
    metric : str
        Metric to compute: 'sns' (sensitivity), 'spe' (specificity), or 'accuracy'.

    Returns
    -------
    score : float
        Computed value for the selected metric.
    """

    # Ensure requested metric is valid
    assert metric in [
        "sns",
        "spe",
        "accuracy",
    ], "Metric must be 'sns', 'spe', or 'accuracy'."

    # Compute confusion matrix
    # [[TN, FP],
    #  [FN, TP]]
    confusion = confusion_matrix(y_true, y_pred)
    TP = confusion[1, 1]  # True positives
    TN = confusion[0, 0]  # True negatives
    FP = confusion[0, 1]  # False positives
    FN = confusion[1, 0]  # False negatives

    # Return metric based on user request
    if metric == "sns":
        # Sensitivity = TP / (TP + FN)
        return TP / float(TP + FN)
    elif metric == "spe":
        # Specificity = TN / (TN + FP)
        return TN / float(TN + FP)
    elif metric == "accuracy":
        # Accuracy = (TP + TN) / total
        return (TP + TN) / float(TP + TN + FP + FN)


# -----------------------------------------------------------------------------
# Function: get_info
# -----------------------------------------------------------------------------
def get_info(model, x_val, y_val, fold_type):
    """
    Compute decision threshold and sensitivity for a validation set.

    This function determines the threshold on predicted probabilities that
    satisfies the predefined specificity (train_specificity). It then computes
    the sensitivity at this threshold.

    Parameters
    ----------
    model : sklearn-like classifier
        Trained model with predict_proba method.
    x_val : pd.DataFrame
        Features of validation set.
    y_val : array-like
        True labels (0 or 1) for validation set.
    fold_type : str
        Fold type for logging/debug purposes ('train' or 'valid').

    Returns
    -------
    list
        [cut_off_threshold, sensitivity]
    """

    # Predicted probabilities for the positive class
    pred_val = model.predict_proba(x_val)[:, 1]

    # Probabilities corresponding to negative/control samples
    control_pred = pred_val[y_val == 0]

    try:
        # Determine threshold that satisfies target specificity
        # Take the minimum probability above the quantile for train_specificity
        cut_off = min(
            control_pred[control_pred > np.quantile(control_pred, train_specificity)]
        )
    except ValueError:
        # Fallback: if no values satisfy, take the quantile directly
        cut_off = np.quantile(control_pred, train_specificity)

    # Apply threshold to classify samples
    y_pred = np.array([1 if y > cut_off else 0 for y in pred_val])

    # Compute sensitivity at this threshold
    sns = get_metrices(y_val, y_pred, "sns")

    return [cut_off, sns]


# -----------------------------------------------------------------------------
# Function: cal_sensitivity_at_specificity
# -----------------------------------------------------------------------------
def cal_sensitivity_at_specificity(
    y_true, y_pred_prob, specificity_threshold, intervals
):
    """
    Determine the maximum sensitivity achievable while maintaining specificity >= threshold.

    The function iteratively increases the probability threshold from min to max
    predicted values, computing specificity and sensitivity at each step, until
    the specificity meets or exceeds the given threshold.

    Parameters
    ----------
    y_true : array-like
        True labels (0 or 1)
    y_pred_prob : array-like
        Predicted probabilities for the positive class.
    specificity_threshold : float
        Desired minimum specificity in [0, 1].
    intervals : int
        Number of steps to increment the threshold between min and max probabilities.

    Returns
    -------
    pred_specificity : float
        Achieved specificity at selected threshold.
    pred_sensitivity : float
        Sensitivity at the threshold.
    threshold : float
        Probability threshold achieving the specificity constraint.
    """

    # Initialize achieved specificity
    pred_specificity = 0

    # Step size for threshold increments
    step = (y_pred_prob.max() - y_pred_prob.min()) / intervals

    # Start threshold slightly above the minimum predicted probability
    k = y_pred_prob.min() + step

    # Iterate until specificity meets or exceeds the target
    while pred_specificity < specificity_threshold:

        if k < y_pred_prob.max():
            # Apply threshold to get predicted labels
            y_pred = [1 if y >= k else 0 for y in y_pred_prob]

            # Compute TN, FP, FN, TP
            tn, fp, fn, tp = confusion_matrix(
                y_true.tolist(), y_pred, labels=[0, 1]
            ).ravel()

            # Compute specificity and sensitivity
            pred_specificity = tn / (tn + fp)
            pred_sensitivity = tp / (tp + fn)

        else:
            # Last step: use max probability as threshold
            k = y_pred_prob.max()
            y_pred = [1 if y >= k else 0 for y in y_pred_prob]
            tn, fp, fn, tp = confusion_matrix(
                y_true.tolist(), y_pred, labels=[0, 1]
            ).ravel()
            pred_specificity = tn / (tn + fp)
            pred_sensitivity = tp / (tp + fn)
            break

        # Increment threshold
        k += step

    # Return the achieved specificity, corresponding sensitivity, and threshold
    return pred_specificity, pred_sensitivity, k
