# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: GPL-3.0-or-later
# =============================================================================
# # -*- coding: utf-8 -*-
# =============================================================================
# SCRIPT  : config.py
# PROJECT : MAPLE - Cancer/Non-cancer Classifier
# PURPOSE : Configuration settings for training and evaluation of the
#           LightGBM-based classifier with Bayesian optimization.
#
# AUTHOR  : Liyuan Zhao
# CREATED : 2025-07-10
# UPDATED : 2025-10-10
# =============================================================================

"""
Configuration module for MAPLE boost-tree classifier.

Contains default hyperparameters, training thresholds, and model metadata.
Modify these parameters to adjust model behavior, training, and evaluation.
"""

# -----------------------------
# Training configuration
# -----------------------------
# Target specificity for training: used to threshold predictions
train_specificity: float = 0.95

# Maximum number of boosting iterations (trees)
# max_boost_round: int = 1000

# Learning rate for LightGBM
# learning_rate: float = 0.05

# Number of folds for cross-validation
# cv_folds: int = 5

# Random seed for reproducibility
# random_seed: int = 42

# -----------------------------
# Model metadata
# -----------------------------
# Model name identifier
NAME: str = "lightGBM_bayesian"

# Description
MODEL_DESCRIPTION: str = (
    "LightGBM gradient boosting model optimized with Bayesian search. "
    "Predicts cancer/non-cancer status based on enriched methylation haplotype features."
)

# -----------------------------
# Notes
# -----------------------------
# - Adjust train_specificity to control false positive rate.
# - max_boost_round, learning_rate, and early_stopping_rounds can be tuned.
# - All paths are relative; modify if needed.
