# =============================================================================
# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: GPL-3.0-or-later
# =============================================================================
# SCRIPT  : MAPLE_select_best_probe.py
# PROJECT : MAPLE (Methylation-Anchor Probe for Low Enrichment)
# PURPOSE : Filter BLASTN-evaluated probes and generate the final probe pool
#
# OVERVIEW:
#   This script integrates thermodynamic and BLASTN evaluation results to classify
#   probes into quality categories (Good, Risk, Remove). It applies multi-step
#   filtering logic to retain only the most specific and stable probes, and then
#   selects the optimal probe per haplotype-chain for final capture pool design.
#
# INPUTS  :
#   - Annotated probe table with thermodynamic and BLASTN metrics (*.txt)
#
# OUTPUTS :
#   - Final probe selection table (*.txt)
#   - Summary report of probe quality categories and filtering statistics
#
# AUTHOR  : Yangjunyi Li
# CREATED : 2023-08-01
# UPDATED : 2025-07-31
#
# NOTE    :
#   - Developed as part of the MAPLE short-probe design pipeline.
#   - Implements hierarchical filtering based on specificity, ΔG, GC%, and Tm.
#   - Produces a finalized probe set optimized for methylation haplotype capture.
# =============================================================================
"""
Script Description:
    This script processes thermodynamically and BLAST-evaluated probes,
    assigns quality labels (Good, Risk, Remove), and selects the best probe
    for each haplotype-Chain based on predefined filtering rules.

Usage:
    python MAPLE_select_best_probe.py --data input_blast_results.txt --output output_prefix

Arguments:
    --data      Path to annotated + BLASTed probe table (tab-delimited .txt)
    --output    Output file prefix for best probe selection
"""
import argparse
import time

import pandas as pd


def filtering(df):
    """
    Filter and select top probe per Chain based on multiple criteria.

    Sorting priority:
    1. blast_label (ascending)
    2. Chain (descending)
    3. GC_content (descending)
    4. deltaG (ascending)
    5. Tm (descending)
    6. length (descending)

    Parameters:
        df (pd.DataFrame): Input DataFrame containing probe information.

    Returns:
        pd.DataFrame: Filtered DataFrame with best probe per Chain.
    """
    sort_order = {
        "blast_label": True,
        "Chain": False,
        "GC_content": False,
        "deltaG": True,
        "Tm": False,
        "length": False,
    }

    sorted_df = df.sort_values(
        by=list(sort_order.keys()), ascending=list(sort_order.values())
    )

    top_probes = sorted_df.groupby("Chain", as_index=False).first()

    return top_probes


def label_blast_res(df):
    """
    Label probes based on BLAST alignment results.

    Rules:
        - 'Good':  Perfect hits <= 1 and Other hits <= 1
        - 'Risk':  Perfect hits <= 1 and 1 < Other hits <= 100
        - 'Remove': Perfect hits > 1 or Other hits > 100

    Parameters:
        df (pd.DataFrame): DataFrame with BLAST hit statistics

    Returns:
        pd.DataFrame: Updated DataFrame with 'blast_label' column
    """
    # Initialize label
    df["blast_label"] = "None"

    # Assign labels based on rules
    df.loc[
        (df["Counts_Perfect_Hits"] <= 1) & (df["Counts_Other_Hits"] <= 1), "blast_label"
    ] = "Good"
    df.loc[
        (df["Counts_Perfect_Hits"] <= 1) & (df["Counts_Other_Hits"].between(2, 100)),
        "blast_label",
    ] = "Risk"
    df.loc[
        (df["Counts_Perfect_Hits"] > 1) | (df["Counts_Other_Hits"] > 100), "blast_label"
    ] = "Remove"

    # Report haplotype counts by category
    def unique_haplos(label):
        return set(df[df["blast_label"] == label]["ID"].str.split("--").str[0])

    good_haplo = unique_haplos("Good")
    risk_haplo = unique_haplos("Risk") - good_haplo
    remove_haplo = unique_haplos("Remove") - risk_haplo - good_haplo

    print("Good Haplo Counts: ", len(good_haplo))
    print("Risk Haplo Counts: ", len(risk_haplo))
    print("Remove Haplo Counts: ", len(remove_haplo))

    return df


def process(data, output):
    """
    Process probe data, apply BLAST labeling and filtering to select best probe per Chain.

    Parameters:
        data (str): Path to input probe table (.txt, tab-separated)
        output (str): Prefix for output file

    Output:
        Writes a tab-separated result file: <output>_best_probe.txt
    """
    # Load raw input
    raw_df = pd.read_csv(data, sep="\t")

    # Parse Haplo and Chain information
    raw_df["Haplo_ID"] = raw_df["ID"].str.split("--").str[0]
    raw_df["Chromosome"] = raw_df["Haplo_ID"].str.extract(r"(^[^:]+)")
    raw_df["Haplo_start"] = raw_df["Haplo_ID"].str.extract(r":(\d+)-")[0]
    raw_df["Haplo_end"] = raw_df["Haplo_ID"].str.extract(r"-(\d+)$")[0]
    raw_df["Chain"] = raw_df["ID"].str.split("--").str[0:2].str.join("--")

    # Label probes with blast quality
    label_df = label_blast_res(raw_df)

    # Filter and select best probe per Chain (excluding "Remove")
    df_filter = filtering(label_df[label_df["blast_label"] != "Remove"])

    # Rename columns for clarity
    df_filter = df_filter.rename(
        columns={"seq": "Template_seq", "qseq": "Query_seq", "sseq": "Subject_seq"}
    )

    # Select columns for final output
    columns_to_export = [
        "Chromosome",
        "Haplo_start",
        "Haplo_end",
        "Haplo_ID",
        "Chain",
        "Template_seq",
        "probe_seq",
        "probe_ID",
        "deltaG",
        "Tm",
        "cpg_cover",
        "cpg_pos",
        "GC_content",
        "Hairpin_Tm",
        "Hairpin_deltaG",
        "HomoDimer_Tm",
        "HomoDimer_deltaG",
        "Counts_Perfect_Hits",
        "Counts_Other_Hits",
        "Max_Tm_Hits",
        "Query_seq",
        "Subject_seq",
        "Filter_tag",
        "blast_label",
    ]

    # Write output
    df_filter[columns_to_export].to_csv(
        f"{output}_best_probe.txt", sep="\t", index=False
    )


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", help="data", type=str)
    parser.add_argument("--output", help="output prefix", type=str)
    args = parser.parse_args()

    ts_init = time.time()
    process(data=args.data, output=args.output)
    print("Done")
    print("Total Time Used:", time.time() - ts_init)
