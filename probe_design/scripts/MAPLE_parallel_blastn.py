# =============================================================================
# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: GPL-3.0-or-later
# =============================================================================
# SCRIPT  : MAPLE_parallel_blastn.py
# PROJECT : MAPLE (Methylation-Anchor Probe for Low Enrichment)
# PURPOSE : Parallelized BLASTN-based off-target screening for designed probes
#
# OVERVIEW:
#   This script performs BLASTN/BLASTN-short alignments for each filtered probe
#   against a methylated or unmethylated reference database to assess specificity.
#   It calculates perfect/non-perfect hit counts, estimates melting temperatures
#   for potential off-target duplexes, and outputs a summary table for probe ranking.
#
# INPUTS  :
#   - Filtered probe table (*.txt, tab-delimited)
#   - Reference FASTA / BLAST database
#
# OUTPUTS :
#   - Probe-level BLASTN results with columns:
#       [Counts_Perfect_Hits, Counts_Other_Hits, Max_Tm_Hits, qseq, sseq]
#   - Saved as: <prefix>_blastn_res.txt
#
# AUTHOR  : Yangjunyi Li
# CREATED : 2023-07-01
# UPDATED : 2025-09-10
#
# NOTE    :
#   - Designed as part of the MAPLE probe-design pipeline for methylation-based
#     cancer early detection research.
#   - Compatible with both methylated and unmethylated genome references.
# =============================================================================

"""
Description:
    Perform BLASTN alignment on filtered probe tables to identify off-target hits.
    Calculates perfect and non-perfect match counts and estimates melting temperatures
    for off-target sequences to evaluate probe specificity.

Usage:
    python MAPLE_parallel_blastn.py --sub_table filtered_probes.txt --reference genome.fa

Arguments:
    --sub_table    Input filtered probe table file (tab-delimited)
    --reference    Reference fasta file or BLAST database (default: genome_mfa.CT_GA_conversion_merged.fa)
"""

import time
import argparse

import numpy as np
import pandas as pd

from io import StringIO
from Bio.SeqUtils import MeltingTemp as MT
from Bio.Blast.Applications import NcbiblastnCommandline


def calc_tm(sequence, dnac1, dnac2, Na, Mg, dNTPs, c_seq):
    """
    Calculate the melting temperature (Tm) of a DNA duplex using the Nearest-Neighbor method.

    Parameters
    ----------
    sequence : str
        DNA sequence (5' → 3') of one strand.
    dnac1 : float
        Concentration of the strand with the higher concentration [nM].
    dnac2 : float
        Concentration of the strand with the lower concentration [nM].
    Na : float
        Monovalent cation concentration (e.g., Na+) [mM].
    Mg : float
        Divalent cation concentration (e.g., Mg2+) [mM].
    dNTPs : float
        dNTP concentration [mM]; affects Mg2+ availability.
    c_seq : str
        Complementary sequence (3' → 5') to `sequence`.

    Returns
    -------
    float
        Estimated melting temperature (°C), or np.nan if the input is invalid or calculation fails.
    """
    try:
        return MT.Tm_NN(
            sequence,
            dnac1=dnac1,
            dnac2=dnac2,
            Na=Na,
            Mg=Mg,
            dNTPs=dNTPs,
            c_seq=c_seq,
        )
    except ValueError:
        return np.nan


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


def run_blastn(id, sequence, reference, pidentity, qcov_hsp_perc=90, threads=1):
    """
    Run BLASTN on a given sequence and parse results.

    Parameters:
    -----------
    id : str
        Sequence identifier.
    sequence : str
        Query nucleotide sequence.
    reference : str
        Reference BLAST database path.
    pidentity : float
        Minimum percent identity for matches.
    qcov_hsp_perc : int, optional
        Query coverage percentage for HSP, default is 90.
    threads : int, optional
        Number of threads for BLAST, default is 1.

    Returns:
    --------
    tuple:
        count_perfect (int): Number of perfect matches (100% identity, full length, zero mismatches).
        count_nonperfect (int): Number of non-perfect matches.
        qseq (str): Query sequence of the best non-perfect hit by bitscore.
        sseq (str): Subject sequence of the best non-perfect hit by bitscore.
        If no hits, returns (0, 0, "AAAA", "AAAA").
    """
    from Bio.Blast.Applications import NcbiblastnCommandline
    from io import StringIO
    import pandas as pd

    fasta = f">{id}\n{sequence}"
    length = len(sequence)

    blast_cols = [
        "qseqid",
        "sseqid",
        "pident",
        "length",
        "mismatch",
        "gapopen",
        "qstart",
        "qend",
        "sstart",
        "send",
        "evalue",
        "bitscore",
        "qcovhsp",
        "qseq",
        "sseq",
        "sstrand",
    ]
    blast_dtypes = {
        "qseqid": str,
        "sseqid": str,
        "pident": float,
        "length": int,
        "mismatch": int,
        "gapopen": int,
        "qstart": int,
        "qend": int,
        "sstart": int,
        "send": int,
        "evalue": str,
        "bitscore": float,
        "qcovhsp": float,
        "qseq": str,
        "sseq": str,
        "sstrand": str,
    }

    blast_fmt = '"6 {}"'.format(" ".join(blast_cols))
    task = "blastn-short" if length <= 40 else "blastn"

    blastn_cline = NcbiblastnCommandline(
        db=reference,
        outfmt=blast_fmt,
        qcov_hsp_perc=qcov_hsp_perc,
        perc_identity=pidentity,
        dust="no",
        soft_masking=False,
        task=task,
        num_threads=threads,
    )

    out, err = blastn_cline(stdin=fasta)
    blast_df = pd.read_table(StringIO(out), names=blast_cols, dtype=blast_dtypes)

    if blast_df.empty:
        return 0, 0, "AAAA", "AAAA"

    perfect_mask = (
        (blast_df.pident == 100)
        & (blast_df.length == length)
        & (blast_df.mismatch == 0)
    )
    perfect_hits = blast_df[perfect_mask]
    count_perfect = len(perfect_hits)
    count_nonperfect = len(blast_df) - count_perfect

    nonperfect_hits = blast_df[~perfect_mask]

    if not nonperfect_hits.empty:
        best_nonperfect = nonperfect_hits.sort_values("bitscore", ascending=False).iloc[
            0
        ]
    else:
        best_nonperfect = blast_df.sort_values("bitscore", ascending=False).iloc[0]

    return count_perfect, count_nonperfect, best_nonperfect.qseq, best_nonperfect.sseq


def process(
    sub_table,
    reference,
    pidentity=90,
    qcov_hsp_perc=90,
    dnac1=250,
    dnac2=250,
    Na=50,
    Mg=0,
    dNTPs=0,
):
    """
    Calculate off-target hits for filtered probes by running BLASTN and computing Tm of hits.

    Parameters:
    -----------
    sub_table : str
        Path to filtered probe table (tab-delimited).
    reference : str
        BLAST database reference fasta.
    pidentity : int, optional
        Minimum percent identity for BLAST hits (default 90).
    qcov_hsp_perc : int, optional
        Query coverage percentage for HSP (default 90).
    dnac1 : int or float, optional
        Concentration of higher concentrated strand [nM] (default 250).
    dnac2 : int or float, optional
        Concentration of lower concentrated strand [nM] (default 250).
    Na : int or float, optional
        Monovalent cation concentration [mM] (default 50).
    Mg : int or float, optional
        Divalent cation concentration [mM] (default 0).
    dNTPs : int or float, optional
        Concentration of dNTPs [mM] (default 0).

    Returns:
    --------
    None, saves results to disk as tab-delimited file with suffix '_blastn_res.txt'.
    """
    import time

    print("========= Calculating off-target hits =========")
    filtered_table = pd.read_csv(sub_table, sep="\t")
    prefix = sub_table.split("_4blast")[0]

    # Initialize columns
    filtered_table["blastn_res"] = np.nan
    filtered_table["Counts_Perfect_Hits"] = np.nan
    filtered_table["Counts_Other_Hits"] = np.nan
    filtered_table["Max_Tm_Hits"] = np.nan

    print("------ Running blastn ")
    ts = time.time()

    # Run blastn for each probe and store results
    filtered_table["blastn_res"] = filtered_table.apply(
        lambda row: run_blastn(
            row["probe_ID"],
            row["probe_seq"],
            reference=reference,
            pidentity=pidentity,
            qcov_hsp_perc=qcov_hsp_perc,
            threads=1,
        ),
        axis=1,
    )

    # Extract info from blastn results tuple
    filtered_table["Counts_Perfect_Hits"] = filtered_table["blastn_res"].str[0]
    filtered_table["Counts_Other_Hits"] = filtered_table["blastn_res"].str[1]
    filtered_table["qseq"] = filtered_table["blastn_res"].str[2]
    filtered_table["sseq"] = filtered_table["blastn_res"].str[3]

    print("------ Getting Hits complementary sequences ")
    filtered_table["c_seq"] = filtered_table["sseq"].apply(
        lambda s: get_rc_seq(s, reverse=False)
    )

    print("------ Calculating Hits Tm ")
    filtered_table["Max_Tm_Hits"] = filtered_table.apply(
        lambda row: calc_tm(row["qseq"], dnac1, dnac2, Na, Mg, dNTPs, row["c_seq"]),
        axis=1,
    )

    print("------ Time used ", time.time() - ts)

    filtered_table.to_csv(f"{prefix}_blastn_res.txt", sep="\t", index=False)


if __name__ == "__main__":
    import argparse

    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Run BLASTN to calculate off-target hits for filtered probes."
    )
    parser.add_argument(
        "--reference",
        type=str,
        default="genome_mfa.CT_GA_conversion_merged.fa",
        help="Reference fasta file or BLAST database (default: genome_mfa.CT_GA_conversion_merged.fa)",
    )
    parser.add_argument(
        "--sub_table",
        type=str,
        required=True,
        help="Input filtered probe table for BLASTN (tab-delimited)",
    )
    # Uncomment to enable parallel processing support
    # parser.add_argument(
    #     "--core",
    #     type=int,
    #     default=16,
    #     help="Number of CPU cores for parallel computing (default: 16)",
    # )

    args = parser.parse_args()

    # Optional: initialize parallel backend if implemented
    # pandarallel.initialize(progress_bar=True, nb_workers=args.core)

    # Run main processing function
    process(sub_table=args.sub_table, reference=args.reference)
