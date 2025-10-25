# =============================================================================
# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: GPL-3.0-or-later
# =============================================================================
# SCRIPT  : MAPLE_cal_haplo_fraction.py
# PROJECT : MAPLE (Methylation-Anchor Probe for Low Enrichment)
# PURPOSE : Compute haplotype-specific fragment ratios from coverage data
#
# OVERVIEW:
#   This script calculates per-locus haplotype ratios based on on-target fragment
#   counts and haplotype-specific coverage derived from MAPLE stitched fragments.
#   It merges coverage and haplotype metadata, fills missing values, and outputs
#   a summary table with the fraction of each haplotype.
#
# INPUTS  :
#   - <sample>.on_target.coverage : Total on-target fragment counts
#   - <sample>.haplo.coverage     : Haplotype-specific fragment counts
#   - Stitch meta table (*.txt)    : Defines haplotype patterns and coordinates
#
# OUTPUTS :
#   - <sample>_haplo_frequency.txt : Per-locus haplotype fraction table
#
# USAGE   :
#   python MAPLE_cal_haplo_fraction.py --sample_name SAMPLE --meta stitch_meta.txt
#
# AUTHOR  : Yangjunyi Li  (liyangjunyi@bytedance.com)
# CREATED : 2023-03-01
# UPDATED : 2025-09-10
#
# NOTE    :
#   - Designed to be called after MAPLE_haplo_fraction.sh for final haplotype ratio computation.
# =============================================================================

import argparse
import pandas as pd


def main(sample_name, stitch_meta_path):
    # Load metadata
    meta = pd.read_csv(
        stitch_meta_path,
        sep="\t",
        dtype=str,
        names=[
            "Chromosome",
            "Start",
            "End",
            "Haplo_Pattern",
            "name_pool",
            "stitch_pattern_OT",
            "stitch_pattern_OB",
        ],
    )

    # Load coverage files
    overall_coverage_path = f"{sample_name}.on_target.coverage"
    haplo_coverage_path = f"{sample_name}.haplo.coverage"

    overall_coverage = pd.read_csv(overall_coverage_path, sep="\t").rename(
        columns={"Target_counts": f"{sample_name}_Target_counts"}
    )
    haplo_coverage = pd.read_csv(haplo_coverage_path, sep="\t").rename(
        columns={"Haplo_counts": f"{sample_name}_Haplo_counts"}
    )

    # Ensure coordinate columns are strings
    for df in [overall_coverage, haplo_coverage]:
        df[["Chromosome", "Start", "End"]] = df[["Chromosome", "Start", "End"]].astype(
            str
        )

    # Merge haplotype-level coverage
    df_mer_haplo = pd.merge(
        meta,
        haplo_coverage,
        on=["Chromosome", "Start", "End", "stitch_pattern_OT", "stitch_pattern_OB"],
        how="left",
    )

    # Merge total coverage
    df_mer_all = pd.merge(
        df_mer_haplo,
        overall_coverage,
        on=["Chromosome", "Start", "End", "Haplo_Pattern"],
        how="left",
    )

    # Fill missing values with 0 and convert to float
    df_mer_all[f"{sample_name}_Target_counts"] = (
        df_mer_all[f"{sample_name}_Target_counts"].astype(float).fillna(0)
    )
    df_mer_all[f"{sample_name}_Haplo_counts"] = (
        df_mer_all[f"{sample_name}_Haplo_counts"].astype(float).fillna(0)
    )

    # Compute ratio
    df_mer_all[f"{sample_name}_Haplot_ratio"] = (
        df_mer_all[f"{sample_name}_Haplo_counts"]
        / df_mer_all[f"{sample_name}_Target_counts"]
    )

    # Save results
    output_file = f"{sample_name}_haplo_frequency.txt"
    df_mer_all.drop_duplicates().to_csv(output_file, sep="\t", index=False)

    print(f"Finished ratio calculation! Output saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute haplotype probe ratio based on coverage files."
    )
    parser.add_argument(
        "--sample_name",
        required=True,
        help="Sample name prefix used to locate .on_target.coverage and .haplo.coverage files",
    )
    parser.add_argument("--meta", required=True, help="Stitch meta table file")

    args = parser.parse_args()
    main(sample_name=args.sample_name, stitch_meta_path=args.meta)
