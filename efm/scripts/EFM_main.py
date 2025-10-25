# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: GPL-3.0-or-later
# =============================================================================
# -*- coding: utf-8 -*-
# =============================================================================
# SCRIPT  : EFM_main.py
# PROJECT : MAPLE (Methylation-Anchor Probe for Low Enrichment)
# PURPOSE : Perform Enrichment Factor Modeling (EFM) to quantify probe enrichment
#           efficiency and compare MAPLE vs Conventional probe performance.
#
# OVERVIEW:
#   This script loads simulated or experimental haplotype capture data, performs
#   median trend analysis, linear and nonlinear fitting between MAPLE and
#   Conventional measurements, generates plots, and saves summarized fitting results.
#
# INPUTS  :
#   - Haplo capture datasets (simulated or experimental)
#   - Functions from EFMFunc.py for data generation and EFK calculation
#
# OUTPUTS :
#   - PDF plots:
#       - Linear regression: Conventional vs Expected
#       - Nonlinear regression: MAPLE vs Conventional
#   - CSV summary tables with experimental enrichment factor (EFK) results
#
# USAGE   :
#   python EFM_main.py
#
# AUTHOR  : Nina Guanyi Xie
# CREATED : 2025-07-10
# UPDATED : 2025-10-10
# =============================================================================

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress
from scipy import optimize
from sklearn.metrics import r2_score
from EFMFunc import f_sim, generate_fit_df, calculate_experimental_efk_temp

# -----------------------------
# Step 1: Generate individual dataframes for hyper- and hypo-methylated haplotypes
# -----------------------------
# cpg_count: number of CpG sites per haplotype
cpg_count = 2
df_hyper_maple = generate_fit_df("Maple", "Hyper", cpg_count=cpg_count, read_depths=50)
df_hyper_conv = generate_fit_df(
    "Conventional", "Hyper", cpg_count=cpg_count, read_depths=20
)
df_hypo_maple = generate_fit_df("Maple", "Hypo", cpg_count=cpg_count, read_depths=50)
df_hypo_conv = generate_fit_df(
    "Conventional", "Hypo", cpg_count=cpg_count, read_depths=20
)

# -----------------------------
# Step 2: Rename columns for clarity
# -----------------------------
hyper_maple_cols = (
    ["Haplo_coord", "haplo_pattern", "Capture_chain", "Target_status"]
    + [f"M_{i}%" for i in [0, 1, 2, 5, 10, 15, 20, 40, 60, 80, 85, 90, 95, 98, 99, 100]]
    + ["NaN_count"]
)
hyper_conv_cols = (
    ["Haplo_coord", "haplo_pattern", "Capture_chain", "Target_status"]
    + [f"L_{i}%" for i in [0, 1, 2, 5, 10, 15, 20, 40, 60, 80, 85, 90, 95, 98, 99, 100]]
    + ["NaN_count"]
)

df_hyper_maple.columns = hyper_maple_cols
df_hyper_conv.columns = hyper_conv_cols
df_hypo_maple.columns = hyper_maple_cols
df_hypo_conv.columns = hyper_conv_cols

# -----------------------------
# Step 3: Merge Maple and Conventional datasets for Hyper and Hypo
# -----------------------------
merge_cols = ["Haplo_coord", "haplo_pattern", "Capture_chain", "Target_status"]
df_final_hyper = pd.merge(
    df_hyper_maple, df_hyper_conv, on=merge_cols, how="inner"
).reset_index(drop=True)
df_final_hypo = pd.merge(
    df_hypo_maple, df_hypo_conv, on=merge_cols, how="inner"
).reset_index(drop=True)

# -----------------------------
# Step 4: Select relevant columns for fitting
# -----------------------------
fit_keys = ["Haplo_coord", "haplo_pattern", "Capture_chain", "Target_status"] + [
    f"_{i}%" for i in [0, 1, 2, 5, 10, 15, 20, 40, 60, 80, 85, 90, 95, 98, 99, 100]
]

df_fit_hyper = df_final_hyper[
    df_final_hyper.columns[df_final_hyper.columns.str.contains("|".join(fit_keys))]
]
df_fit_hypo = df_final_hypo[
    df_final_hypo.columns[df_final_hypo.columns.str.contains("|".join(fit_keys))]
]

# -----------------------------
# Step 5: Perform nonlinear fit and save plots to PDF
# -----------------------------
output_filename_hyper = f"2506_Hyper_{cpg_count}.pdf"
output_hyper = calculate_experimental_efk_temp(df_fit_hyper, output_filename_hyper)

output_filename_hypo = f"2506_Hypo_{cpg_count}.pdf"
output_hypo = calculate_experimental_efk_temp(df_fit_hypo, output_filename_hypo)

# Save fitting results as CSV
output_hyper.to_csv(f"summary_fit_results_Hyper_{cpg_count}.csv", index=False)
output_hypo.to_csv(f"summary_fit_results_Hypo_{cpg_count}.csv", index=False)

# -----------------------------
# Step 6: Median trend analysis between MAPLE (M) and Conventional (L)
# -----------------------------
fit_pattern = [
    f"_{i}%" for i in [0, 1, 2, 5, 10, 15, 20, 40, 60, 80, 85, 90, 95, 98, 99, 100]
]

select_columns = df_fit_hyper.columns[
    df_fit_hyper.columns.str.contains("|".join(fit_pattern))
]
df_median_hyper = df_fit_hyper[select_columns]

# Separate MAPLE and Conventional columns
df_maple = df_median_hyper.loc[:, df_median_hyper.columns.str.contains("M_")]
df_conv = df_median_hyper.loc[:, df_median_hyper.columns.str.contains("L_")]

# Calculate medians per titration column
median_maple = df_maple.median(axis=0).values.tolist()
median_conv = df_conv.median(axis=0).values.tolist()

# Expected fractions for titrations (0-100%)
expected_vals = [
    0.00,
    0.01,
    0.02,
    0.05,
    0.10,
    0.15,
    0.20,
    0.40,
    0.60,
    0.80,
    0.85,
    0.90,
    0.95,
    0.98,
    0.99,
    1.00,
]

# -----------------------------
# Step 7: Linear regression: Conventional vs Expected
# -----------------------------
x_conv = np.array(median_conv, dtype=float)
y_exp = np.array(expected_vals, dtype=float)

slope, intercept, r_value, _, _ = linregress(x_conv, y_exp)
y_pred_lin = slope * x_conv + intercept

print(
    f"[Linear fit] Slope: {slope:.3f}, Intercept: {intercept:.3f}, R²: {r_value**2:.3f}"
)

plt.figure(figsize=(5, 5))
plt.scatter(x_conv, y_exp, label="Data points")
plt.plot(
    x_conv,
    y_pred_lin,
    color="red",
    label=f"Fit: y = {slope:.2f}x + {intercept:.2f}, R² = {r_value**2:.3f}",
)
plt.ylim([-0.1, 1.1])
plt.xlim([-0.1, 1.1])
plt.xlabel("Median Observed haplotype (Conventional)")
plt.ylabel("Expected haplotype fraction")
plt.legend(loc="upper left", ncol=1, frameon=True)
plt.title("Linear fit of Conventional vs Expected")
plt.tight_layout()
plt.savefig("fit_conventional_vs_expected.pdf")

# -----------------------------
# Step 8: Nonlinear regression: MAPLE vs Conventional
# -----------------------------
obs_maple = np.array(median_maple, dtype=float)
true_conv = np.array(median_conv, dtype=float)

# Fit MAPLE to Conventional data using nonlinear function f_sim
popt, _ = optimize.curve_fit(f_sim, obs_maple, true_conv)
k = np.round(popt[0], 2)
x_curve = np.linspace(0, 1, 100)
y_curve = f_sim(x_curve, k)

y_pred_nonlinear = f_sim(obs_maple, *popt)
r2_nl = np.round(r2_score(true_conv, y_pred_nonlinear), 3)

print(f"[Nonlinear fit] k: {k}, R²: {r2_nl}")

plt.figure(figsize=(5, 5))
plt.plot(obs_maple, true_conv, "o", label="Captured %")
plt.plot(x_curve, y_curve, label=f"Fit line: k = {k}, R² = {r2_nl}", color="green")
plt.xlabel("Median MAPLE haplotype frequency")
plt.ylabel("Median Conventional haplotype frequency")
plt.title("Nonlinear fit: MAPLE vs Conventional")
plt.legend(loc="upper left", ncol=1, frameon=True)
plt.tight_layout()
plt.savefig("fit_maple_vs_conventional.pdf")
