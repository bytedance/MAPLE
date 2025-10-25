# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: GPL-3.0-or-later
# =============================================================================
# -*- coding: utf-8 -*-
# =============================================================================
# SCRIPT  : EFMFunc.py
# PROJECT : MAPLE (Methylation-Anchor Probe for Low Enrichment)
# PURPOSE : Provide utility functions for Enrichment Factor Modeling (EFM)
#           including curve fitting, haplotype matrix processing, and
#           calculation of experimental enrichment factors (k values and R²).
#
# FUNCTIONS PROVIDED:
#   - f_sim: Simulated nonlinear function for enrichment curve fitting
#   - return_matrix_haplo: Load and filter haplotype matrices
#   - generate_fit_df: Merge haplotype matrices across titrations
#   - calculate_experimental_efk_temp: Fit observed vs. true fractions, output PDF plots
#
# AUTHOR  : Nina Guanyi Xie
# CREATED : 2025-07-10
# UPDATED : 2025-10-10
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy import optimize
from sklearn.metrics import r2_score


# -----------------------------
# Nonlinear simulation function for enrichment fitting
# -----------------------------
def f_sim(f2, k):
    """
    Simulated function for nonlinear enrichment curve fitting.

    Parameters
    ----------
    f2 : array-like
        Observed haplotype fraction (MAPLE or other method).
    k : float
        Enrichment factor parameter.

    Returns
    -------
    np.ndarray
        Predicted true fraction after applying enrichment factor k.
    """
    f2 = np.array(f2)
    return f2 / (f2 + k - k * f2)


# -----------------------------
# Load and filter a single haplotype matrix
# -----------------------------
def return_matrix_haplo(file_name, target_pattern, wt_pattern, read_depths):
    """
    Extract and filter haplotype matrix from a file based on target methylation
    pattern and minimum read depth.

    Parameters
    ----------
    file_name : str
        Path to haplotype fraction file.
    target_pattern : str
        Target methylation pattern to retain (e.g., 'M,M,M').
    wt_pattern : str
        Wild-type methylation pattern (used for filtering if needed).
    read_depths : int
        Minimum read depth threshold.

    Returns
    -------
    pd.DataFrame
        Filtered haplotype DataFrame.
    """
    # Extract sample name from file
    sample_name = file_name.split("_")[1]

    # Load CSV and drop unnecessary columns
    df = pd.read_csv(file_name, sep="\t").drop("Sample", axis=1)

    # Filter by read depth
    df = df[df["groupsum"] > read_depths]

    # Keep relevant columns and rename
    df = df[
        [
            "Haplo_coord",
            "Haplo_status",
            "haplo_pattern",
            "Capture_chain",
            "Target_status",
            "proportion",
        ]
    ]
    df = df.rename(columns={"proportion": f"proportion_{sample_name}"})

    # Filter rows matching target methylation pattern
    df_final = df[
        (df["Target_status"] == target_pattern) & (df["Haplo_status"] == target_pattern)
    ].reset_index(drop=True)

    # Drop redundant columns
    df_final = df_final.drop(columns=["Haplo_status"])
    return df_final


# -----------------------------
# Merge haplotype matrices across titration series
# -----------------------------
def generate_fit_df(method, methyl_target, cpg_count, read_depths):
    """
    Combine haplotype matrices across multiple titrations into a single DataFrame.

    Parameters
    ----------
    method : str
        Method name ('Maple' or 'Conventional').
    methyl_target : str
        'Hyper' or 'Hypo' methylation target.
    cpg_count : int
        Number of CpG sites per haplotype.
    read_depths : int
        Minimum read depth threshold.

    Returns
    -------
    pd.DataFrame
        Combined haplotype DataFrame ready for fitting.
    """
    # Define target and wild-type patterns
    target_pattern = (
        ",".join(["M"] * cpg_count)
        if methyl_target == "Hyper"
        else ",".join(["U"] * cpg_count)
    )
    wt_pattern = (
        ",".join(["U"] * cpg_count)
        if methyl_target == "Hyper"
        else ",".join(["M"] * cpg_count)
    )

    # Predefined titration fractions
    fraction_ID = [0, 1, 2, 5, 10, 15, 20, 40, 60, 80, 85, 90, 95, 98, 99, 100]

    df_final = None
    for i, frac in enumerate(fraction_ID):
        file_name = (
            f"{'maple' if method == 'Maple' else 'conv'}_{frac}_haplotype_fraction.txt"
        )
        df_temp = return_matrix_haplo(
            file_name, target_pattern, wt_pattern, read_depths
        )
        # Merge across titrations
        df_final = (
            df_temp
            if i == 0
            else pd.merge(
                df_final,
                df_temp,
                on=["Haplo_coord", "haplo_pattern", "Capture_chain", "Target_status"],
                how="inner",
            )
        )

    # Remove rows with missing values
    df_final["NaN_count"] = df_final.isna().sum(axis=1)
    df_final = df_final[df_final["NaN_count"] <= 0]
    return df_final


# -----------------------------
# Compute enrichment factor k and R² for each haplotype
# -----------------------------
def calculate_experimental_efk_temp(df_input, filename):
    """
    Fit observed vs. true fractions and compute experimental enrichment factors.

    Parameters
    ----------
    df_input : pd.DataFrame
        Input dataframe containing observed and true haplotype fractions.
    filename : str
        Output PDF filename to store plots.

    Returns
    -------
    pd.DataFrame
        DataFrame with haplotype, Capture_chain, k value, and R² for each haplotype.
    """
    r_square_list, k_list, status_list, haplotype_list, chain_list = [], [], [], [], []

    plots_per_page = 24
    fig, axes, plot_count = None, None, 0

    with PdfPages(filename) as pdf:
        for _, row in df_input.iterrows():
            # Extract observed and true fractions
            obs_v = np.array(row[4:20].tolist())
            true_v = np.array(row[20:].tolist())
            mask = ~np.isnan(obs_v) & ~np.isnan(true_v)
            obs_v_clean, true_v_clean = obs_v[mask], true_v[mask]

            if len(obs_v_clean) <= 1:
                continue

            # Fit nonlinear enrichment function
            popt, _ = optimize.curve_fit(f_sim, obs_v_clean, true_v_clean)
            k = np.round(popt, 2)
            y_pred = f_sim(obs_v_clean, *popt)
            r_squared = np.round(r2_score(true_v_clean, y_pred), 3)

            # Initialize new page if needed
            if plot_count % plots_per_page == 0:
                if fig:
                    pdf.savefig(fig)
                    plt.close(fig)
                fig, axes = plt.subplots(6, 4, figsize=(15, 20))
                axes = axes.flatten()

            ax = axes[plot_count % plots_per_page]
            ax.plot(obs_v_clean, true_v_clean, "o", label="Captured %")
            ax.plot(
                np.linspace(0, 1, 100),
                f_sim(np.linspace(0, 1, 100), k),
                label=f"k = {k}, R² = {r_squared}",
            )
            ax.set_title(f"{row['haplo_pattern']} + {row['Capture_chain']}")
            ax.legend()
            plot_count += 1

            # Append results
            r_square_list.append(r_squared)
            k_list.append(*popt)
            haplotype_list.append(row["haplo_pattern"])
            chain_list.append(row["Capture_chain"])
            status_list.append(row["Target_status"])

        # Save last figure
        if fig:
            pdf.savefig(fig)
            plt.close(fig)

    return pd.DataFrame(
        {
            "Haplo_status": status_list,
            "haplo_pattern": haplotype_list,
            "Capture_chain": chain_list,
            "k": k_list,
            "RSquare": r_square_list,
        }
    )
