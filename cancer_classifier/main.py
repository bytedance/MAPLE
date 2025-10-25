# =============================================================================
# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: GPL-3.0-or-later
# =============================================================================
# SCRIPT  : main.py
# PROJECT : LightGBM-based Bayesian Optimized Classifier
# PURPOSE : Train, optimize, and evaluate a binary classifier with stratified sampling.
#
# OVERVIEW:
#   Reads input tab-delimited data with features and metadata, splits data into
#   training and testing sets, performs cross-validation with Bayesian
#   hyperparameter optimization, trains the final LightGBM model, and evaluates
#   performance on validation and test sets.
#
# INPUTS  :
#   - <input_data>.tsv : Tab-delimited file containing features and metadata,
#                        including a 'Label' column ('Healthy' or 'CRC') and
#                        optional stratification columns.
#
# OUTPUTS :
#   - <output_dir>/temp_data/ : Intermediate train/test data files.
#   - <output_dir>/model/     : Trained LightGBM model and best parameters.
#   - <output_dir>/results/   : Predictions and performance metrics.
#
# USAGE   :
#   python main.py -i <input_data> -o <output_dir> [options]
#
# AUTHOR  : Liyuan Zhao
# CREATED : 2025-10-10
# UPDATED : 2025-10-10
#
# NOTE    :
#   - Requires Python >= 3.8, pandas, numpy, scikit-learn, lightgbm, bayes_opt.
#   - Designed for binary classification tasks with optional stratified splits.
# =============================================================================

import argparse
import os
from pathlib import Path
import logging
import warnings

import pandas as pd
import numpy as np
from lightgbm import log_evaluation
from sklearn.metrics import roc_auc_score

from config import train_specificity, NAME
from utils.logger import init_logger
from utils.io import save_data, save_json, save_model
from data_utils.split import train_test_split_func, cv_split_func
from model.train import lgb_LGBMClassifier, nested_cv_lgm, evaluate_on_cv
from model.evaluate import get_metrices, cal_sensitivity_at_specificity

# Suppress warnings for cleaner logs
warnings.filterwarnings("ignore")


# =============================================================================
# Function: parse_args
# =============================================================================
def parse_args():
    """
    Parse command line arguments for input/output files, sampling, and model options.

    Returns
    -------
    argparse.Namespace : Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        prog=f"{NAME}",
        formatter_class=argparse.RawTextHelpFormatter,
        description=f"{NAME}",
    )

    # Input and output parameters
    parser.add_argument(
        "-i",
        "--input_data",
        type=str,
        metavar="PATH",
        help="Input tab-delimited data file with features and metadata",
        required=True,
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        metavar="PATH",
        help="Output directory to save results and model",
        required=True,
    )

    # Test set sampling options
    parser.add_argument(
        "--test_sampling_method",
        type=str,
        default="stratified",
        choices=["stratified", "random"],
        help="Sampling method for test set (default: stratified)",
    )
    parser.add_argument(
        "--test_sampling_metric",
        type=str,
        help="Column name used for stratified test split",
    )
    parser.add_argument(
        "--test_sampling_frac",
        type=float,
        default=0.8,
        help="Fraction of data for test set (default: 0.8)",
    )

    # Cross-validation options
    parser.add_argument(
        "--validation_sampling_method",
        type=str,
        default="random",
        choices=["stratified", "random"],
        help="Sampling method for cross-validation (default: random)",
    )
    parser.add_argument(
        "--validation_sampling_metric",
        type=str,
        help="Column name used for stratified CV split",
    )
    parser.add_argument(
        "--validation_sampling_fold",
        type=int,
        default=5,
        help="Number of folds for cross-validation (default: 5)",
    )
    parser.add_argument(
        "--optimize_times",
        type=int,
        default=10,
        help="Number of repeats for Bayesian optimization (default: 10)",
    )

    # Seed for reproducibility
    parser.add_argument(
        "--sampling_seed",
        type=int,
        default=132,
        help="Random seed for train/test and CV splits (default: 132)",
    )

    return parser.parse_args()


# =============================================================================
# Function: main
# =============================================================================
def main():
    """
    Main workflow:
    1. Parse command-line arguments
    2. Prepare output directories and logging
    3. Load and preprocess data
    4. Split data into train/test
    5. Generate CV splits
    6. Optimize hyperparameters using Bayesian optimization
    7. Train final model and evaluate on validation and test sets
    8. Save models, parameters, and predictions
    """
    args = parse_args()

    # Extract arguments
    input_data = args.input_data
    output_dir = Path(args.output_dir)
    test_sampling_method = args.test_sampling_method
    test_sampling_frac = args.test_sampling_frac
    test_sampling_metric = args.test_sampling_metric
    validation_sampling_method = args.validation_sampling_method
    validation_sampling_fold = args.validation_sampling_fold
    validation_sampling_metric = args.validation_sampling_metric
    random_state = args.sampling_seed
    max_repeat_times = args.optimize_times

    # Create required output directories
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "temp_data").mkdir(exist_ok=True)
    (output_dir / "model").mkdir(exist_ok=True)
    (output_dir / "results").mkdir(exist_ok=True)

    # Initialize logging
    init_logger(output_dir)
    logging.info("-" * 60)
    logging.info(f"> Input data                  : {input_data}")
    logging.info(f"> Output directory            : {output_dir}")
    logging.info(f"> Sampling seed               : {random_state}")
    logging.info(f"> Test sampling method        : {test_sampling_method}")
    logging.info(f"> Test sampling metric        : {test_sampling_metric}")
    logging.info(f"> Test sampling fraction      : {test_sampling_frac}")
    logging.info(f"> Validation sampling method  : {validation_sampling_method}")
    logging.info(f"> Validation sampling metric  : {validation_sampling_metric}")
    logging.info(f"> Validation folds            : {validation_sampling_fold}")
    logging.info(f"> Bayesian optimize repeats   : {max_repeat_times}")
    logging.info("-" * 60)

    # ------------------------------
    # Load and preprocess data
    # ------------------------------
    logging.info("##### Data Processing #####")
    all_data = pd.read_csv(input_data, sep="\t", header=0, index_col=0)
    all_data = all_data.sort_values(by="sampleID").reset_index(
        drop=True
    )  # consistent sorting
    logging.info(f"> Number of samples: {all_data.shape[0]}")

    if test_sampling_metric:
        dist_df = pd.DataFrame(
            all_data[test_sampling_metric].value_counts()
        ).reset_index()
        dist_df = dist_df.sort_values(by="index").reset_index(drop=True)
        logging.info(f"> Distribution of {test_sampling_metric}: \n{dist_df}")

    # ------------------------------
    # Train/Test split
    # ------------------------------
    logging.info("##### Train/Test Split #####")
    train_data_all, train_data, train_meta, test_data_all, test_data, test_meta = (
        train_test_split_func(
            all_data,
            sampling=test_sampling_method,
            test_size=test_sampling_frac,
            metric=test_sampling_metric,
            random_state=random_state,
        )
    )
    save_data(train_data_all, output_dir / "temp_data" / "train_data.csv")
    save_data(test_data_all, output_dir / "temp_data" / "test_data.csv")

    # ------------------------------
    # Generate CV splits
    # ------------------------------
    logging.info("##### Cross-validation Split #####")
    cv_split = list(
        cv_split_func(
            train_data,
            train_meta,
            sampling=validation_sampling_method,
            metric=validation_sampling_metric,
            fold=validation_sampling_fold,
            random_state=random_state,
        )
    )

    # ------------------------------
    # Bayesian Optimization
    # ------------------------------
    logging.info("##### Bayesian Optimization #####")
    callbacks = [log_evaluation(period=10000)]
    best_params = nested_cv_lgm(
        train_data_all,
        train_meta,
        max_repeat_times,
        cv_split,
        callbacks,
        train_specificity,
    )
    save_json(best_params, output_dir / "model" / "best_params.json")
    logging.info(f"> Best hyperparameters: {best_params}")

    # ------------------------------
    # Evaluate on CV
    # ------------------------------
    logging.info("##### Train & Evaluate CV #####")
    valid_fold_res = evaluate_on_cv(train_data_all, train_meta, best_params, cv_split)
    sns_cv, spe_cv, optimal_cutoff = cal_sensitivity_at_specificity(
        valid_fold_res.Label.replace({"CRC": 1, "Healthy": 0}),
        valid_fold_res["pred_prob"],
        train_specificity,
        10000,
    )
    logging.info(f"> Optimized cutoff: {optimal_cutoff}")

    # Apply threshold to assign predicted labels
    valid_fold_res["pred_label"] = np.where(
        valid_fold_res["pred_prob"] > optimal_cutoff, 1, 0
    )
    save_data(valid_fold_res, output_dir / "results" / "train_set_pred.csv")
    logging.info(f"> Validation set Sensitivity: {round(sns_cv, 4)}")
    logging.info(f"> Validation set Specificity: {round(spe_cv, 4)}")

    # ------------------------------
    # Train final model on full training set
    # ------------------------------
    logging.info("##### Train Final Model #####")
    final_model = lgb_LGBMClassifier(best_params)
    final_model.fit(train_data, train_meta["Label"].replace({"Healthy": 0, "CRC": 1}))
    save_model(final_model, output_dir / "model" / "model.dat")

    # ------------------------------
    # Predict and evaluate on test set
    # ------------------------------
    logging.info("##### Test Set Prediction #####")
    pred_prob = final_model.predict_proba(test_data)[:, 1]
    y_pred = np.where(pred_prob > optimal_cutoff, 1, 0)

    sns_test = get_metrices(
        test_meta["Label"].replace({"Healthy": 0, "CRC": 1}), y_pred, metric="sns"
    )
    spe_test = get_metrices(
        test_meta["Label"].replace({"Healthy": 0, "CRC": 1}), y_pred, metric="spe"
    )
    auc_test = roc_auc_score(
        test_meta["Label"].replace({"Healthy": 0, "CRC": 1}), pred_prob
    )

    logging.info(f"> Test set Sensitivity: {round(sns_test, 4)}")
    logging.info(f"> Test set Specificity: {round(spe_test, 4)}")
    logging.info(f"> Test set AUC        : {round(auc_test, 4)}")

    test_meta["pred_prob"] = pred_prob
    test_meta["pred_label"] = y_pred
    test_meta["cutoff"] = optimal_cutoff
    save_data(test_meta, output_dir / "results" / "test_set_pred.csv")

    logging.info(f"##### Workflow completed! Results saved to: {output_dir} #####")


# Entry point
if __name__ == "__main__":
    main()
