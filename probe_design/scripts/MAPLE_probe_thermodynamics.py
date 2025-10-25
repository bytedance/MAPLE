# =============================================================================
# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: GPL-3.0-or-later
# =============================================================================
# SCRIPT  : MAPLE_probe_thermodynamics.py
# PROJECT : MAPLE (Methylation-Anchor Probe for Low Enrichment)
# PURPOSE : Calculate thermodynamic parameters and perform probe-level QC evaluation
#
# OVERVIEW:
#   This script computes key thermodynamic properties for each designed probe,
#   including melting temperature (Tm), GC content, and Gibbs free energy (ΔG).
#   It applies predefined QC thresholds to filter probes suitable for capture design,
#   ensuring optimal hybridization stability and uniform capture efficiency.
#
# INPUTS  :
#   - Probe sequence table (*.txt, tab-delimited)
#   - User-defined thresholds for ΔG, GC%, probe length, and temperature conditions
#
# OUTPUTS :
#   - Probe-level thermodynamic QC table (*.txt) with annotated properties:
#       [Probe_ID, GC%, Tm, ΔG, Pass_QC]
#   - Summary statistics for downstream probe selection
#
# AUTHOR  : Yangjunyi Li
# CREATED : 2023-07-01
# UPDATED : 2025-09-10
#
# NOTE    :
#   - Developed as part of the MAPLE short-probe design pipeline.
#   - Supports parameter tuning for various capture and hybridization conditions.
#   - Compatible with downstream modules for off-target analysis and ML-based scoring.
# =============================================================================
"""
Script Description:
    This script calculates thermodynamic features (e.g., melting temperature, GC content, 
    free energy) for each probe and evaluates quality control (QC) metrics 
    necessary for short probe design.

Usage:
    python MAPLE_probe_thermodynamics.py \
        --deltaG <min> <max> \
        --gc <min> <max> \
        --probe_length <length> \
        --prefix <input_file_prefix> \
        --output <output_file_prefix> \
        [--tm <min_melting_temp>] \
        [--salinity <monovalent_cation_mM>] \
        [--dnac1 <strand1_conc_nM>] \
        [--dnac2 <strand2_conc_nM>] \
        [--temp <reaction_temp_C>] \
        [--dntp <dNTP_conc_mM>]

Arguments:
    --deltaG (float float, required): Minimum and maximum ΔG (free energy) range for filtering probes, e.g., -10.0 -5.0
    --gc (float float, required): Minimum and maximum GC content percentage range, e.g., 40.0 60.0
    --probe_length (int, required): Length of probes to analyze (step size)
    --prefix (str, required): Input file prefix for probe sequences (expects files named like <prefix>_<probe_length>.txt)
    --output (str, required): Output file prefix to save QC results
    --tm (float, optional, default=60): Minimum melting temperature in Celsius (default 60°C)
    --salinity (float, optional, default=50): Monovalent cation concentration in mM (default 50 mM)
    --dnac1 (float, optional, default=250): Concentration of the higher concentrated strand in nM (default 250 nM)
    --dnac2 (float, optional, default=250): Concentration of the lower concentrated strand in nM (default 250 nM)
    --temp (float, optional, default=50): Reaction temperature in Celsius (default 50°C)
    --dntp (float, optional, default=0): dNTP concentration in mM (default 0 mM)
"""

import time
import argparse


import pandas as pd
import numpy as np
from itertools import groupby
import primer3
from Bio.SeqUtils import MeltingTemp as MT


def create_deltaG_array(salinity, temp):
    """
    Generate a ΔG (delta G) array for DNA duplex stability calculation using the Nearest-Neighbor model.

    Parameters
    ----------
    salinity : float
        Monovalent cation concentration in millimolar (mM), e.g., Na+ or K+.
    temp : float
        Reaction temperature in degrees Celsius.

    Returns
    -------
    np.ndarray
        A 4x4 numpy array of ΔG values (kcal/mol) corresponding to [A, T, C, G] × [A, T, C, G] base pairs.
    """
    # Convert salinity from mM to M
    salinity /= 1000.0
    # Convert temperature from Celsius to Kelvin
    temp_K = temp + 273.15

    # Enthalpy values (ΔH, in kcal/mol) for each dinucleotide step
    paraH = np.array(
        [
            [-7.6, -7.2, -8.4, -7.8],  # A
            [-7.2, -7.6, -8.2, -8.5],  # T
            [-8.5, -7.8, -8.0, -10.6],  # C
            [-8.2, -8.4, -9.8, -8.0],  # G
        ]
    )

    # Entropy values (ΔS, in cal/mol·K) for each dinucleotide step
    paraS = np.array(
        [
            [-21.3, -20.4, -22.4, -21.0],  # A
            [-21.3, -21.3, -22.2, -22.7],  # T
            [-22.7, -21.0, -19.9, -27.2],  # C
            [-22.2, -22.4, -24.4, -19.9],  # G
        ]
    )

    # Adjust entropy based on ionic strength (SantaLucia correction)
    paraS += 0.368 * np.log(salinity)

    # Calculate free energy: ΔG = ΔH - T·ΔS/1000 (converted to kcal/mol)
    paraG = paraH - (temp_K * paraS / 1000.0)

    return paraG


def calc_deltaG(sequence, salinity, temp):
    """
    Estimate the Gibbs free energy (ΔG) of a DNA sequence using the Nearest-Neighbor model.

    Parameters
    ----------
    sequence : str
        DNA sequence composed of characters A, C, T, G.
    salinity : float
        Monovalent cation concentration in millimolar (mM), e.g., [Na+].
    temp : float
        Temperature of the reaction in degrees Celsius.

    Returns
    -------
    float
        Estimated ΔG (in kcal/mol) of the input sequence.
        Returns np.nan if the sequence contains invalid characters.
    """
    # Generate the nearest-neighbor ΔG matrix for the given condition
    paraG = create_deltaG_array(salinity, temp)

    # Convert temperature to Kelvin for calculation
    temp_K = temp + 273.15

    # Initialization energy (SantaLucia 1998 model)
    dG = 0.2 - temp_K * (-5.7) / 1000  # In kcal/mol

    # Base mapping: A=0, T=1, C=2, G=3
    mapping = {"A": 0, "T": 1, "C": 2, "G": 3}

    # Loop through adjacent base pairs
    for base1, base2 in zip(sequence, sequence[1:]):
        i = mapping.get(base1.upper())
        j = mapping.get(base2.upper())
        if i is None or j is None:
            return np.nan  # Invalid base found
        dG += paraG[i, j]

    return dG


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


def get_cg(seq_id):
    """
    Extract CpG coverage and position on a probe from the sequence ID.

    Parameters
    ----------
    seq_id : str
        Formatted sequence ID containing genomic CpG position and probe info.

    Returns
    -------
    cpg_info : str
        CpG positions on the probe (5' → 3') and their IDs, formatted as:
        "pos1::pos2::...--id1::id2::..."
    cpg_count : int
        Number of CpGs covered by the probe.
    """
    # Extract CpG positions and IDs
    cg_pos = seq_id.split("--")[2].split("::")  # 0-based index (from pandas)
    cg_id = seq_id.split("_")[0].split("--")[3].split("::")

    # Extract probe range (from seqkit, 1-based)
    seq_pos = seq_id.split("_sliding:")[-1].split("-")
    start, end = int(seq_pos[0]), int(seq_pos[1])
    probe_length = end - start + 1

    probe_cg_id = []
    probe_cg_pos = []

    if len(cg_pos) == len(cg_id):
        for i in range(len(cg_pos)):
            cg = int(cg_pos[i])
            # Check if CpG falls within probe region (adjusting for 0/1-based)
            if (cg >= start - 1) and (cg + 1 <= end - 1):
                # Calculate position on reverse-complement probe
                pos_on_probe = probe_length - (cg - (start - 1)) - 1
                probe_cg_id.append(cg_id[i])
                probe_cg_pos.append(pos_on_probe)

        # Reverse to match probe's 5' → 3' direction
        probe_cg_id_reverse = list(reversed(probe_cg_id))
        probe_cg_pos_reverse = list(reversed(probe_cg_pos))

        return (
            "::".join(map(str, probe_cg_pos_reverse))
            + "--"
            + "::".join(probe_cg_id_reverse),
            len(probe_cg_id_reverse),
        )
    else:
        return "", 0


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


def calc_secondary(sequence, temp, Na, Mg, dnac1, dNTPs):
    """
    Calculate secondary structure thermodynamics (hairpin and homodimer) for a probe sequence.

    Parameters
    ----------
    sequence : str
        Probe DNA sequence (should be < 60 bp for accurate prediction).
    temp : float
        Reaction temperature in degrees Celsius.
    Na : float
        Monovalent cation concentration [mM].
    Mg : float
        Divalent cation concentration [mM].
    dnac1 : float
        Concentration of the higher-concentration strand [nM].
    dNTPs : float
        dNTP concentration [mM].

    Returns
    -------
    hairpin_tm : float
        Melting temperature (°C) of predicted hairpin structure.
    hairpin_dg : float
        Gibbs free energy (ΔG, kcal/mol) of hairpin formation.
    homodimer_tm : float
        Melting temperature (°C) of predicted homodimer.
    homodimer_dg : float
        Gibbs free energy (ΔG, kcal/mol) of homodimer formation.

    Notes
    -----
    - Returns NaNs if sequence length is ≥ 60 nt.
    - ΔG is converted from cal/mol to kcal/mol by dividing by 1000.
    """
    if len(sequence) >= 60:
        print("⚠️ Secondary structure cannot be predicted for probes ≥ 60 bp.")
        return np.nan, np.nan, np.nan, np.nan

    res_hairpin = primer3.calc_hairpin(
        sequence,
        mv_conc=Na,
        dv_conc=Mg,
        dntp_conc=dNTPs,
        dna_conc=dnac1,
        temp_c=temp,
        max_loop=30,
        output_structure=False,
    )
    res_homodimer = primer3.calc_homodimer(
        sequence,
        mv_conc=Na,
        dv_conc=Mg,
        dntp_conc=dNTPs,
        dna_conc=dnac1,
        temp_c=temp,
        max_loop=30,
        output_structure=False,
    )

    return (
        res_hairpin.tm,
        res_hairpin.dg / 1000,
        res_homodimer.tm,
        res_homodimer.dg / 1000,
    )


def check_restriction_site(seq, restri_sites=["GTATCC", "GGATAC"]):
    """
    Check whether a sequence contains any specified restriction sites.

    Parameters
    ----------
    seq : str
        Probe sequence to scan.
    restri_sites : list of str, optional
        List of restriction site sequences to check for.

    Returns
    -------
    str
        Concatenated string of restriction sites found in the sequence.
        Returns an empty string if none are found.
    """
    found_sites = [site for site in restri_sites if site in seq]
    return "".join(found_sites)


def check_polyX(seq, n=10):
    """
    Check if the sequence contains any homopolymer stretches (PolyX).

    Parameters
    ----------
    seq : str
        Probe sequence to scan.
    n : int, optional
        Minimum number of repeated bases to qualify as a homopolymer.

    Returns
    -------
    str
        A comma-separated string of detected homopolymers (e.g., "PolyA, PolyT").
        Returns an empty string if none are found.
    """
    poly_list = []
    for base, group in groupby(seq):
        run_length = sum(1 for _ in group)
        if run_length >= n:
            poly_list.append(f"Poly{base}")
    return ", ".join(poly_list)


def annotation(row, deltaG, Tm_min, GC_content, hairpin_tm=45, homodimer_tm=45):
    """
    Annotate the raw probe DataFrame with given constraints.

    Parameters
    ----------
    row : pd.Series
        A row from the probe stats DataFrame (e.g., df.loc[i]).
    deltaG : list or tuple
        [min_dG, max_dG], in kcal/mol.
    Tm_min : float
        Minimum acceptable melting temperature.
    GC_content : list or tuple
        [min_GC, max_GC], in percent.
    hairpin_tm : float, optional
        Threshold for flagging hairpin risk (default: 45).
    homodimer_tm : float, optional
        Threshold for flagging homodimer risk (default: 45).

    Returns
    -------
    str
        A string of tags indicating which constraints were violated.
        May also include detected PolyX or restriction sites.
    """
    filter_tag = ""
    cpg_haplo = len(
        row["ID"].split("--")[2].split("::")
    )  # probe must fully cover haplotype

    if float(row["cpg_cover"]) < cpg_haplo:
        filter_tag += "less_CpG,"
    if float(row["deltaG"]) < deltaG[0]:
        filter_tag += "low_dg,"
    if float(row["deltaG"]) > deltaG[1]:
        filter_tag += "high_dg,"
    if float(row["Tm"]) < Tm_min:
        filter_tag += "low_Tm,"
    if float(row["Hairpin_Tm"]) > hairpin_tm:
        filter_tag += "hairpin_risk,"
    if float(row["HomoDimer_Tm"]) > homodimer_tm:
        filter_tag += "homodimer_risk,"
    if float(row["GC_content"]) < GC_content[0]:
        filter_tag += "low_GC,"
    if float(row["GC_content"]) > GC_content[1]:
        filter_tag += "high_GC,"

    ployx = check_polyX(row["probe_seq"])
    rsite = check_restriction_site(row["probe_seq"])
    return filter_tag + ployx + rsite


def probes_thermo(input_table, temp, dnac1, dnac2, dNTPs, Na, Mg=0, c_seq=None):
    print("========= Fetching target chains =========")
    ts = time.time()

    print(f"----- Loading input table: {input_table}")
    seq_table = pd.read_csv(input_table, sep="\t", header=None)
    seq_table.columns = ["ID", "seq", "length"]
    print(f"------ Time used: {time.time() - ts:.2f} seconds")

    ts = time.time()
    print("------ Generating probe sequences (reverse complement)")
    seq_table["probe_seq"] = seq_table["seq"].apply(get_rc_seq)
    print(f"------ Time used: {time.time() - ts:.2f} seconds")

    # Initialize columns for thermodynamic properties
    seq_table["probe_ID"] = np.nan
    seq_table["deltaG"] = np.nan
    seq_table["Tm"] = np.nan
    seq_table["cpg_cover"] = np.nan
    seq_table["cpg_pos"] = np.nan
    seq_table["GC_content"] = np.nan
    seq_table["secondary"] = np.nan
    seq_table["Hairpin_Tm"] = np.nan
    seq_table["Hairpin_deltaG"] = np.nan
    seq_table["HomoDimer_Tm"] = np.nan
    seq_table["HomoDimer_deltaG"] = np.nan

    print("========= Calculating Thermodynamics =========")

    # Calculate deltaG
    print("------ Calculating deltaG")
    ts = time.time()
    seq_table["deltaG"] = seq_table["probe_seq"].apply(
        lambda seq: calc_deltaG(seq, Na, temp)
    )
    print(f"------ Time used: {time.time() - ts:.2f} seconds")

    # Calculate melting temperature (Tm)
    print("------ Calculating Tm")
    ts = time.time()

    def calc_tm_row(row):
        return calc_tm(
            row["probe_seq"],
            dnac1=dnac1,
            dnac2=dnac2,
            Na=Na,
            Mg=Mg,
            dNTPs=dNTPs,
            c_seq=get_rc_seq(row["probe_seq"], reverse=False),
        )

    seq_table["Tm"] = seq_table.apply(calc_tm_row, axis=1)
    print(f"------ Time used: {time.time() - ts:.2f} seconds")

    # Calculate CpG coverage and positions
    print("------ Calculating probe CpG coverage and positions")
    ts = time.time()
    seq_table["cpg_cover"] = seq_table["ID"].apply(lambda id_: get_cg(id_)[1])
    seq_table["cpg_pos"] = seq_table["ID"].apply(lambda id_: get_cg(id_)[0])
    seq_table["probe_ID"] = seq_table["ID"] + "---" + seq_table["cpg_pos"]
    print(f"------ Time used: {time.time() - ts:.2f} seconds")

    # Calculate GC content
    print("------ Calculating probe GC content")
    ts = time.time()
    seq_table["GC_content"] = seq_table["probe_seq"].apply(cal_gc_content)
    print(f"------ Time used: {time.time() - ts:.2f} seconds")

    # Predict secondary structure properties
    print("------ Predicting secondary structure")
    ts = time.time()
    seq_table["secondary"] = seq_table.apply(
        lambda row: calc_secondary(
            row["probe_seq"], temp=temp, Na=Na, Mg=Mg, dnac1=dnac1, dNTPs=dNTPs
        ),
        axis=1,
    )

    # Only unpack secondary structure data if length < 60 (prediction limit)
    unique_lengths = seq_table["length"].unique()
    if len(unique_lengths) == 1 and unique_lengths[0] < 60:
        seq_table["Hairpin_Tm"] = seq_table["secondary"].str[0]
        seq_table["Hairpin_deltaG"] = seq_table["secondary"].str[1]
        seq_table["HomoDimer_Tm"] = seq_table["secondary"].str[2]
        seq_table["HomoDimer_deltaG"] = seq_table["secondary"].str[3]
    else:
        print("Secondary structure prediction skipped for sequences >= 60 bp")

    print(f"------ Time used: {time.time() - ts:.2f} seconds")
    return seq_table


def process(
    deltaG,
    Tm_min,
    GC_content,
    step,
    prefix,
    output,
    temp,
    dnac1,
    dnac2,
    Na,
    dNTPs,
    Mg=0,
):
    # Run thermodynamic calculations on probes
    raw_table = probes_thermo(
        prefix + f"_{step}.txt",
        temp=temp,
        dnac1=dnac1,
        dnac2=dnac2,
        Na=Na,
        Mg=Mg,
        dNTPs=dNTPs,
    )

    print("========= Annotation & Filtering =========")
    ts = time.time()

    # Annotate probes with filter tags
    raw_table["Filter_tag"] = raw_table.apply(
        lambda row: annotation(
            row, deltaG=deltaG, Tm_min=Tm_min, GC_content=GC_content
        ),
        axis=1,
    )

    # Save full annotated table
    raw_table.to_csv(f"{output}_{step}_probe_annotated.txt", sep="\t", index=False)

    # Filter probes that pass all filters (empty Filter_tag)
    filtered_table = raw_table[raw_table["Filter_tag"] == ""]
    filtered_table.to_csv(f"{output}_{step}_probe_filtered.txt", sep="\t", index=False)

    print(f"========= Input Probe Number: {len(raw_table)}")
    print(f"========= Probe Number after QC & Thermo Filtering: {len(filtered_table)}")
    print(f"------ Time used: {time.time() - ts:.2f} seconds")

    print("========= Sliding DataFrame for blastn =========")

    # Split filtered table into chunks of 1600 rows for downstream blastn
    chunk_size = 1600
    df_dict = {
        n: filtered_table.iloc[n : n + chunk_size, :]
        for n in range(0, len(filtered_table), chunk_size)
    }

    for i, start_idx in enumerate(df_dict.keys()):
        df_dict[start_idx].to_csv(
            f"{output}_{step}_sub_{i}_probe_4blastn.txt",
            sep="\t",
            index=False,
        )

    print(f"------ Time used: {time.time() - ts:.2f} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Probe thermodynamics and filtering pipeline"
    )

    parser.add_argument(
        "--deltaG", nargs=2, type=float, required=True, help="DeltaG range: min max"
    )
    parser.add_argument(
        "--gc", nargs=2, type=float, required=True, help="GC content range: min max"
    )
    parser.add_argument(
        "--probe_length", type=int, required=True, help="Length of probes (step)"
    )
    parser.add_argument("--prefix", type=str, required=True, help="Input file prefix")
    parser.add_argument(
        "--tm", type=float, default=60, help="Minimum melting temperature (default: 60)"
    )
    parser.add_argument(
        "--salinity",
        type=float,
        default=50,
        help="Monovalent cation concentration [mM] (default: 50)",
    )
    parser.add_argument(
        "--dnac1",
        type=float,
        default=250,
        help="Concentration of higher concentrated strand [nM] (default: 250)",
    )
    parser.add_argument(
        "--dnac2",
        type=float,
        default=250,
        help="Concentration of lower concentrated strand [nM] (default: 250)",
    )
    parser.add_argument(
        "--temp",
        type=float,
        default=50,
        help="Reaction temperature in Celsius (default: 50)",
    )
    parser.add_argument(
        "--dntp", type=float, default=0, help="Concentration of dNTPs [mM] (default: 0)"
    )
    parser.add_argument("--output", type=str, required=True, help="Output prefix")

    args = parser.parse_args()

    ts_init = time.time()
    process(
        deltaG=args.deltaG,
        Tm_min=args.tm,
        GC_content=args.gc,
        step=args.probe_length,
        prefix=args.prefix,
        Na=args.salinity,
        temp=args.temp,
        dnac1=args.dnac1,
        dnac2=args.dnac2,
        dNTPs=args.dntp,
        output=args.output,
    )

    print("Done")
    print(f"Total Time Used: {time.time() - ts_init:.2f} seconds")
