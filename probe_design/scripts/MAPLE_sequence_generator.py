# =============================================================================
# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: GPL-3.0-or-later
# =============================================================================
# SCRIPT  : MAPLE_sequence_generator.py
# PROJECT : MAPLE (Methylation-Anchor Probe for Low Enrichment)
# PURPOSE : Generate strand-specific sequences and haplotype status tables
#
# OVERVIEW:
#   This script processes haplotype tables containing DMR or variant information
#   to reconstruct genomic sequences for both hyper- and hypo-methylated states.
#   It extracts corresponding regions from the reference genome, applies user-defined
#   sequence extensions, and generates chain-specific FASTA and annotation tables
#   for downstream probe sliding and thermodynamic evaluation.
#
# INPUTS  :
#   - Haplotype annotation table (TSV)
#   - Reference genome FASTA file
#
# OUTPUTS :
#   - Extended haplotype sequence tables (*.txt)
#   - Strand-specific FASTA files for hyper/hypo probes
#
#
# AUTHOR  : Yangjunyi Li
# CREATED : 2023-07-01
# UPDATED : 2025-07-31
#
# NOTE    :
#   - Serves as the first module in the MAPLE short-probe design pipeline.
#   - Supports multi-chain generation (e.g., OT/OB, CTOT/CTOB).
#   - Output files feed directly into the probe sliding and thermodynamic stages.
# =============================================================================
"""
Usage:
    python MAPLE_sequence_generator.py --data input_table.txt --reference reference.fa --extend 50 --output_prefix output_prefix

Description:
    This script processes haplotype data to generate DNA chains and haplotype status tables
    used for short probe design. It reads an input table with DMR/Haplotype information,
    extracts sequences from the reference genome, and outputs annotated sequence files.
"""

import re
import argparse
import time

import numpy as np
import pandas as pd
from pyfaidx import Fasta  # For fast fasta file access


def get_ref_seq(chr, start, end, ref):
    """
    Extract a genomic DNA sequence from a specified region of the reference genome.

    Parameters
    ----------
    chr : str
        Chromosome identifier (e.g., 'chr1', '1', 'X')
    start : int
        Start position (0-based, inclusive)
    end : int
        End position (0-based, exclusive)
    ref : pyfaidx.Fasta
        Indexed reference genome loaded via pyfaidx

    Returns
    -------
    str
        Uppercase DNA sequence corresponding to the specified genomic interval
    """
    ref_seq = ref[str(chr)][int(start) : int(end)]

    # Ensure the sequence is uppercase; print region if unexpected format occurs
    if not str(ref_seq).isupper():
        print(f"Warning: Sequence not uppercase for region {chr}:{start}-{end}")
        print(ref_seq)

    return str(ref_seq).upper()


def get_target_seq(chr, start, end, ref, extend):
    """
    Extract the reference sequence for a given genomic region with non-CG-flanking extensions.

    The function returns a sequence that includes:
    - The haplotype region of interest
    - Upstream and downstream flanking sequences
    - Flanks are trimmed to exclude CG dinucleotides to avoid methylation bias

    Parameters
    ----------
    chr : str
        Chromosome identifier (e.g., 'chr1', '1', 'X')
    start : int
        Start position of the core haplotype (0-based, inclusive)
    end : int
        End position of the core haplotype (0-based, exclusive)
    ref : pyfaidx.Fasta
        Reference genome loaded using pyfaidx
    extend : int
        Flanking length to extract on both sides

    Returns
    -------
    str
        Uppercase DNA sequence composed of:
        [flank_without_CG] + [haplotype_with_CG] + [flank_without_CG]
    """
    # Extract haplotype region
    hap = get_ref_seq(chr, start, end, ref)

    # Extract upstream flank, mask CG, and keep only the part after the last CG
    head_ext = get_ref_seq(chr, start - extend, start, ref)
    head_ext_noCG = head_ext.replace("CG", "--").split("--")[-1]

    # Extract downstream flank, mask CG, and keep only the part before the first CG
    tail_ext = get_ref_seq(chr, end, end + extend, ref)
    tail_ext_noCG = tail_ext.replace("CG", "--").split("--")[0]

    # Combine non-CG flanks with the CG-containing haplotype
    seq_new = head_ext_noCG + hap + tail_ext_noCG

    return seq_new.upper()


def methy_replace(seq, new_sub, n, sub="--"):
    """
    Replace the N-th occurrence of a specific sub-pattern in a sequence.

    Useful for targeted substitution of CG-masked dinucleotides or other custom placeholders
    (e.g., replacing the N-th "--" with a desired methylation-aware pattern).

    Parameters
    ----------
    seq : str
        The original input sequence.
    new_sub : str
        The string that will replace the N-th occurrence of `sub`.
    n : int
        The index (1-based) of the `sub` occurrence to replace.
    sub : str, optional
        The substring pattern to find and replace (default is "--").

    Returns
    -------
    str
        The modified sequence with the N-th `sub` replaced by `new_sub`.

    Raises
    ------
    IndexError
        If the specified N-th occurrence does not exist in the input sequence.
    """
    matches = [m.start() for m in re.finditer(sub, seq)]

    if n > len(matches) or n < 1:
        raise IndexError(f"Requested occurrence #{n} of '{sub}' not found in sequence.")

    where = matches[n - 1]

    # Reconstruct the sequence with the desired substitution
    before = seq[:where]
    after = seq[where:]
    after = after.replace(
        sub, new_sub, 1
    )  # Replace only first instance in the 'after' part

    return before + after


def get_bisseq_with_pattern(refseq, pattern):
    """
    Simulate bisulfite conversion of a reference sequence based on a haplotype methylation pattern.

    This function converts non-CG cytosines (C) to thymines (T) to simulate bisulfite treatment,
    while selectively replacing CG sites based on the provided methylation pattern.

    Parameters
    ----------
    refseq : str
        DNA sequence from the reference genome, expected to contain CG sites.
    pattern : str
        Haplotype methylation pattern consisting of:
            'M' - methylated CpG (remain as CG)
            'U' - unmethylated CpG (converted to TG)
        The length of this pattern must equal the number of CG sites in `refseq`.

    Returns
    -------
    str
        Simulated bisulfite-converted sequence.

    Raises
    ------
    ValueError
        If the pattern length does not match the number of CG sites,
        or if invalid characters are found in the pattern.
    """
    cg_count = refseq.count("CG")

    if len(pattern) != cg_count:
        raise ValueError(
            f"Mismatch between CG sites ({cg_count}) and pattern length ({len(pattern)})."
        )

    # Mask all CG sites as placeholders to protect them from non-CG conversion
    seq_masked = refseq.replace("CG", "--")

    # Convert all remaining Cs (non-CG) to Ts
    seq_conv = seq_masked.replace("C", "T")

    # Restore CG sites based on methylation pattern
    for status in pattern:
        if status == "U":
            seq_conv = methy_replace(seq_conv, "TG", 1, "--")
        elif status == "M":
            seq_conv = methy_replace(seq_conv, "CG", 1, "--")
        else:
            raise ValueError(
                "Pattern must contain only 'M' (methylated) or 'U' (unmethylated)."
            )

    return seq_conv


def get_rc_seq(sequence, reverse=True):
    """
    Return the reverse complement (or complement only) of a DNA sequence.

    Parameters
    ----------
    sequence : str
        DNA sequence (A, T, C, G).
    reverse : bool, optional
        If True (default), return the reverse complement.
        If False, return the complement in the original order.

    Returns
    -------
    str
        Reverse complement or complement of the input sequence.
    """
    complement = {"A": "T", "T": "A", "C": "G", "G": "C"}

    # Get complement of each base, optionally reversed
    if reverse:
        return "".join(complement.get(base, base) for base in reversed(sequence))
    else:
        return "".join(complement.get(base, base) for base in sequence)


def cal_gc_content(seq):
    """
    Calculate the GC content of a DNA sequence.

    Parameters
    ----------
    seq : str
        DNA sequence (expected to contain characters A, T, C, G).

    Returns
    -------
    float
        GC content as a fraction between 0 and 1.

    Raises
    ------
    ValueError
        If the input sequence is empty.
    """
    if len(seq) == 0:
        raise ValueError("Input sequence is empty, cannot compute GC content.")

    gc_count = seq.count("G") + seq.count("C")
    gc_content = gc_count / len(seq)

    return gc_content


def cal_cpg_distance(seq):
    """
    Calculate CpG site distribution metrics within a DNA sequence.

    Parameters
    ----------
    seq : str
        DNA sequence in which CpG sites ('CG') are analyzed.

    Returns
    -------
    tuple:
        list_dst : list of int
            Distances between consecutive CpG sites (in bases).
        mean_dst : float
            Mean distance between CpG sites (rounded to 2 decimals).
        std_dst : float
            Standard deviation of distances (rounded to 4 decimals).
        max_dst : int
            Maximum distance spanned by all CpG sites.
        dense_dst : float
            Density of CpG sites per base (rounded to 4 decimals).
    """
    # Split sequence at CpG sites to measure distances between them
    # Each segment length +1 approximates distance between CpGs
    list_dst = [len(segment) + 1 for segment in seq.split("CG")[1:-1]]

    mean_dst = round(np.mean(list_dst), 2) if list_dst else 0.0
    std_dst = round(np.std(list_dst), 4) if list_dst else 0.0

    # max_dst approximates total span of CpG distances
    max_dst = np.sum(list_dst) + len(list_dst) - 1 if list_dst else 0

    # CpG density: count of CpGs divided by total sequence length
    dense_dst = round((len(list_dst) + 1) / len(seq), 4) if len(seq) > 0 else 0.0

    return list_dst, mean_dst, std_dst, max_dst, dense_dst


def cpg_pos(seq, chain):
    """
    Get CpG site positions formatted as a string based on the chain type.

    Parameters
    ----------
    seq : str
        DNA sequence to analyze for CpG sites.
    chain : str
        Chain type, determines how CpG positions are reported:
        - "OT", "CTOB", "ref": forward strand positions
        - "OB", "CTOT": reverse strand positions (coordinates flipped)

    Returns
    -------
    str
        Formatted string representing CpG positions and their indices:
        "<pos1>::<pos2>::...--<id1>::<id2>::..."
    """
    # Find start positions of all "CG" occurrences in the sequence
    pos_ref = [m.start() for m in re.finditer("CG", seq)]

    # Position indices from 1 to number of CpGs
    pos_id = list(range(1, len(pos_ref) + 1))

    if chain in {"OT", "CTOB", "ref"}:
        # Forward strand: return positions and IDs as is
        return "::".join(map(str, pos_ref)) + "--" + "::".join(map(str, pos_id))
    elif chain in {"OB", "CTOT"}:
        # Reverse strand: flip positions relative to sequence length
        pos_ref_rev = list(reversed(pos_ref))
        flipped_pos = [len(seq) - pos - 1 for pos in pos_ref_rev]
        flipped_id = list(reversed(pos_id))
        return "::".join(map(str, flipped_pos)) + "--" + "::".join(map(str, flipped_id))
    else:
        raise ValueError(f"Unsupported chain type: {chain}")


def generate_fasta(df, output_prefix):
    """
    Export probe sequences and their IDs to text files for probe design.

    Generates four tab-separated files, each containing ID and sequence columns
    for different chain types: OT, OB, CTOT, and CTOB.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing probe design data with columns:
        'OT_ID', 'OT_chain', 'OB_ID', 'OB_chain', 'CTOT_ID', 'CTOT_chain', 'CTOB_ID', 'CTOB_chain'.
    output_prefix : str
        Prefix for output filenames.
    """
    print(f"Writing probe design files with prefix: {output_prefix}")

    df[["OT_ID", "OT_chain"]].to_csv(
        f"{output_prefix}_OT_4design.txt", sep="\t", index=False, header=False
    )
    df[["OB_ID", "OB_chain"]].to_csv(
        f"{output_prefix}_OB_4design.txt", sep="\t", index=False, header=False
    )
    df[["CTOT_ID", "CTOT_chain"]].to_csv(
        f"{output_prefix}_CTOT_4design.txt", sep="\t", index=False, header=False
    )
    df[["CTOB_ID", "CTOB_chain"]].to_csv(
        f"{output_prefix}_CTOB_4design.txt", sep="\t", index=False, header=False
    )


def calculate_haplo_coord_OT(cpg_stats, start, pattern):
    """
    Calculate haplotype coordinates on the OT (original top) chain based on CpG gap statistics.

    Parameters
    ----------
    cpg_stats : str
        CpG gap distribution string in the format '[gap1,gap2,...]'.
    start : int or str
        Starting coordinate of the first CpG site.
    pattern : str
        Haplotype pattern string, e.g., 'MUMM'.

    Returns
    -------
    str
        Comma-separated string of coordinates with corresponding haplotype states,
        formatted as "pos:state,pos:state,...".

    Raises
    ------
    ValueError
        If the length of CpG gaps does not match the length of the haplotype pattern minus one.
    """
    # Parse gap distances from the cpg_stats string
    gap_list = cpg_stats.split("[")[1].split("]")[0].split(",")

    if len(gap_list) != len(pattern) - 1:
        raise ValueError(
            "Length of CpG gap list does not match haplotype pattern length minus one."
        )

    # Initialize coordinate string with first position and pattern state
    coord = f"{start}:{pattern[0]},"
    current_pos = int(start)

    # Calculate subsequent positions by adding gaps + 1 for each CpG site
    for i in range(1, len(pattern)):
        current_pos += int(gap_list[i - 1]) + 1
        coord += f"{current_pos}:{pattern[i]},"

    return coord.rstrip(",")


def calculate_haplo_coord_OB(cpg_stats, start, pattern):
    """
    Calculate haplotype coordinates on the OB (original bottom) chain based on CpG gap statistics.

    Parameters
    ----------
    cpg_stats : str
        CpG gap distribution string in the format '[gap1,gap2,...]'.
    start : int or str
        Starting coordinate of the first CpG site.
    pattern : str
        Haplotype pattern string, e.g., 'MUMM'.

    Returns
    -------
    str
        Comma-separated string of coordinates with corresponding haplotype states,
        formatted as "pos:state,pos:state,...".

    Raises
    ------
    ValueError
        If the length of CpG gaps does not match the length of the haplotype pattern minus one.
    """
    gap_list = cpg_stats.split("[")[1].split("]")[0].split(",")

    if len(gap_list) != len(pattern) - 1:
        raise ValueError(
            "Length of CpG gap list does not match haplotype pattern length minus one."
        )

    current_pos = int(start)
    # First position is start + 1, per original logic
    coord = f"{current_pos + 1}:{pattern[0]},"

    for i in range(1, len(pattern)):
        current_pos += int(gap_list[i - 1]) + 1
        coord += f"{current_pos + 1}:{pattern[i]},"

    return coord.rstrip(",")


def get_full_table(raw_table, ref, extend):
    """
    Generate a full table with target chain sequences and related metrics
    based on the input DMR/Haplotype raw table.

    Parameters
    ----------
    raw_table : pandas.DataFrame
        Raw DMR/Haplotype table containing columns like 'Chromosome', 'Haplo_start', 'Haplo_end',
        'Haplo_pattern', 'Haplo_ID', 'Methy_diff', etc.
    ref : str
        File path to the reference genome (FASTA format).
    extend : int
        Number of bases to extend on each side when extracting sequences.

    Returns
    -------
    pandas.DataFrame
        Extended dataframe including sequence chains, GC content, CpG stats,
        haplotype stitch patterns, and methylation status.
    """
    reference = Fasta(ref)

    # Initialize columns for haplotype sequences and annotations
    raw_table["Hapseq"] = np.nan
    raw_table["Hapseq_CG_anno"] = np.nan
    raw_table["Hapseq_GC"] = np.nan
    raw_table["Hapseq_cpg_stats"] = np.nan

    # Extract haplotype sequence with correct coordinates (-1 for 0-based indexing)
    raw_table["Hapseq"] = raw_table.apply(
        lambda row: get_ref_seq(
            row["Chromosome"], row["Haplo_start"] - 1, row["Haplo_end"] + 1, reference
        ),
        axis=1,
    ).str.upper()
    raw_table["Hapseq_CG_anno"] = raw_table["Hapseq"].str.replace("CG", "--")
    raw_table["Hapseq_GC"] = raw_table["Hapseq"].apply(cal_gc_content)
    raw_table["Hapseq_cpg_stats"] = (
        raw_table["Hapseq"].apply(cal_cpg_distance).astype(str)
    )

    # Extract extended reference sequence around the haplotype region
    raw_table["refseq"] = raw_table.apply(
        lambda row: get_target_seq(
            row["Chromosome"],
            row["Haplo_start"] - 1,
            row["Haplo_end"] + 1,
            reference,
            extend,
        ),
        axis=1,
    )
    raw_table["refseq_CG_anno"] = raw_table["refseq"].str.replace("CG", "--")
    raw_table["refseq_GC"] = raw_table["refseq"].apply(cal_gc_content)
    raw_table["refseq_cpg_stats"] = (
        raw_table["refseq"].apply(cal_cpg_distance).astype(str)
    )

    # Generate OT chain sequences and metrics
    raw_table["OT_chain"] = raw_table.apply(
        lambda row: get_bisseq_with_pattern(row["refseq"], row["Haplo_pattern"]), axis=1
    ).str.upper()
    raw_table["OT_GC"] = raw_table["OT_chain"].apply(cal_gc_content)
    raw_table["OT_cpg_pos"] = (
        raw_table["refseq"].apply(lambda seq: cpg_pos(seq, "OT")).astype(str)
    )
    raw_table["Haplo"] = raw_table["Haplo_ID"] + "-" + raw_table["Haplo_pattern"]
    raw_table["OT_ID"] = raw_table["Haplo"] + "--OT--" + raw_table["OT_cpg_pos"]

    # Generate OB chain sequences and metrics (reverse complement & pattern)
    raw_table["refseq_rc"] = raw_table["refseq"].apply(get_rc_seq)
    raw_table["pattern_rc"] = raw_table["Haplo_pattern"].str[::-1]
    raw_table["OB_chain"] = raw_table.apply(
        lambda row: get_bisseq_with_pattern(row["refseq_rc"], row["pattern_rc"]), axis=1
    )
    raw_table["OB_GC"] = raw_table["OB_chain"].apply(cal_gc_content)
    raw_table["OB_cpg_pos"] = (
        raw_table["refseq"].apply(lambda seq: cpg_pos(seq, "OB")).astype(str)
    )
    raw_table["OB_ID"] = raw_table["Haplo"] + "--OB--" + raw_table["OB_cpg_pos"]

    # Generate CTOT chain sequences and metrics (reverse complement of OT chain)
    raw_table["CTOT_chain"] = raw_table["OT_chain"].apply(get_rc_seq).str.upper()
    raw_table["CTOT_GC"] = raw_table["CTOT_chain"].apply(cal_gc_content)
    raw_table["CTOT_cpg_pos"] = (
        raw_table["refseq"].apply(lambda seq: cpg_pos(seq, "CTOT")).astype(str)
    )
    raw_table["CTOT_ID"] = raw_table["Haplo"] + "--CTOT--" + raw_table["CTOT_cpg_pos"]

    # Generate CTOB chain sequences and metrics (reverse complement of OB chain)
    raw_table["CTOB_chain"] = raw_table["OB_chain"].apply(get_rc_seq).str.upper()
    raw_table["CTOB_GC"] = raw_table["CTOB_chain"].apply(cal_gc_content)
    raw_table["CTOB_cpg_pos"] = (
        raw_table["refseq"].apply(lambda seq: cpg_pos(seq, "CTOB")).astype(str)
    )
    raw_table["CTOB_ID"] = raw_table["Haplo"] + "--CTOB--" + raw_table["CTOB_cpg_pos"]

    # Calculate stitch patterns for OT and OB chains for downstream analysis
    raw_table["stitch_pattern_OT"] = raw_table.apply(
        lambda row: calculate_haplo_coord_OT(
            row["Hapseq_cpg_stats"], row["Haplo_start"], row["Haplo_pattern"]
        ),
        axis=1,
    )
    raw_table["stitch_pattern_OB"] = raw_table.apply(
        lambda row: calculate_haplo_coord_OB(
            row["Hapseq_cpg_stats"], row["Haplo_start"], row["Haplo_pattern"]
        ),
        axis=1,
    )

    # Add methylation status labels for splitting
    raw_table["Methyl"] = "None"
    raw_table.loc[raw_table["Methy_diff"] > 0, "Methyl"] = "Hyper"
    raw_table.loc[raw_table["Methy_diff"] < 0, "Methyl"] = "Hypo"

    return raw_table


def process(data, extend, ref, output_prefix):
    """
    Main processing pipeline:
    - Reads input data
    - Generates full annotated table with sequences and metrics
    - Saves full table and methylation subset tables
    - Produces FASTA files for probe design

    Parameters
    ----------
    data : str
        Path to the input TSV file containing raw DMR/haplotype data.
    extend : int
        Number of bases to extend when extracting sequences.
    ref : str
        Path to the reference genome FASTA file.
    output_prefix : str
        Prefix for all output files.
    """
    print("======= Fetching Input =======")
    ts = time.time()
    table = pd.read_csv(data, sep="\t")
    print(f"------ Time used: {time.time() - ts:.2f} seconds")

    print("======= Generating Full Table =======")
    ts = time.time()
    full_table = get_full_table(table, ref, extend)
    print(f"------ Time used: {time.time() - ts:.2f} seconds")

    # Save full annotated table
    full_table.to_csv(f"{output_prefix}_fulltable.txt", sep="\t", index=False)

    # Save subsets based on methylation status
    full_table[full_table["Methyl"] == "Hyper"].to_csv(
        f"{output_prefix}_fulltable_hyper.txt", sep="\t", index=False
    )
    full_table[full_table["Methyl"] == "Hypo"].to_csv(
        f"{output_prefix}_fulltable_hypo.txt", sep="\t", index=False
    )

    # Save unique haplotype status information
    haplo_status_cols = [
        "Haplo_ID",
        "Haplo_start",
        "Haplo_end",
        "Haplo_pattern",
        "Haplo_length",
        "Haplo_nCG",
        "Frequency_case",
        "Frequency_control",
        "Frequency_diff",
        "stitch_pattern_OT",
        "stitch_pattern_OB",
    ]
    full_table[haplo_status_cols].drop_duplicates().to_csv(
        f"{output_prefix}_Haplotype_status.txt", sep="\t", index=False
    )

    print("======= Producing fasta for Probe Design =======")
    ts = time.time()
    generate_fasta(
        full_table[full_table["Methyl"] == "Hyper"], f"{output_prefix}_hyper"
    )
    generate_fasta(full_table[full_table["Methyl"] == "Hypo"], f"{output_prefix}_hypo")
    print(f"------ Time used: {time.time() - ts:.2f} seconds")


if __name__ == "__main__":
    """
    Command-line interface for the methylation haplotype processing pipeline.
    """

    parser = argparse.ArgumentParser(
        description="Methylation haplotype sequence processing"
    )
    parser.add_argument(
        "--reference",
        type=str,
        default="hs37d5_J024591_L091372_EBV_HBV_HPV.fasta",
        help="Reference FASTA file (non-converted)",
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Input table file path (TSV format)",
    )
    parser.add_argument(
        "--extend",
        type=int,
        required=True,
        help="Number of bases to extend on both sides when extracting sequences",
    )
    parser.add_argument(
        "--output_prefix",
        type=str,
        required=True,
        help="Prefix for all output files",
    )
    args = parser.parse_args()

    process(
        data=args.data,
        extend=args.extend,
        ref=args.reference,
        output_prefix=args.output_prefix,
    )
