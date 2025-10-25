# =============================================================================
# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: GPL-3.0-or-later
# =============================================================================
# SCRIPT  : io.py
# PROJECT : MAPLE (Methylation-Anchor Probe for Low Enrichment)
# PURPOSE : Input/output utility functions for saving DataFrames, models, and JSON
#
# OVERVIEW:
#   Provides simple, reusable functions to serialize and save pandas DataFrames,
#   machine learning models (via joblib), and dictionaries/lists as JSON files.
#   Ensures consistent file formatting and encoding for downstream MAPLE pipelines.
#
# INPUTS  :
#   - pd.DataFrame : DataFrames to be saved as CSV
#   - model        : Trained ML models for serialization
#   - dict/list    : Python objects to be saved as JSON
#
# OUTPUTS :
#   - CSV files for DataFrames
#   - Joblib binary files for models
#   - JSON files for dict/list objects
#
# USAGE   :
#   save_data(df, "output.csv")
#   save_model(model, "model.pkl")
#   save_json(data_dict, "config.json")
#
# AUTHOR  : Liyuan Zhao
# CREATED : 2025-07-10
# UPDATED : 2025-10-10
#
# NOTE    :
#   - Overwrites existing files at the same path.
#   - JSON files are saved with UTF-8 encoding and pretty formatting.
# =============================================================================

import pandas as pd
import json
import joblib
from typing import Any, Union


# -----------------------------------------------------------------------------
# Function: save_data
# -----------------------------------------------------------------------------
def save_data(data: pd.DataFrame, path: str) -> None:
    """
    Save a pandas DataFrame to a CSV file.

    Parameters
    ----------
    data : pd.DataFrame
        The DataFrame to be saved.
    path : str
        File path to save the CSV.

    Notes
    -----
    - Saves with header and without row indices.
    - Overwrites existing files at the same path.
    """
    data.to_csv(path, index=False, header=True)


# -----------------------------------------------------------------------------
# Function: save_model
# -----------------------------------------------------------------------------
def save_model(model: Any, path: str) -> None:
    """
    Serialize and save a machine learning model using joblib.

    Parameters
    ----------
    model : Any
        Trained model object (e.g., sklearn, LightGBM, or custom object).
    path : str
        File path to save the serialized model.

    Notes
    -----
    - Saves the model in binary format compatible with joblib.load().
    """
    joblib.dump(model, path)


# -----------------------------------------------------------------------------
# Function: save_json
# -----------------------------------------------------------------------------
def save_json(data: Union[dict, list], path: str) -> None:
    """
    Save a dictionary or list as a JSON file.

    Parameters
    ----------
    data : dict or list
        Data structure to be serialized as JSON.
    path : str
        File path to save the JSON.

    Notes
    -----
    - Overwrites existing files at the same path.
    - Uses standard JSON formatting.
    """
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
