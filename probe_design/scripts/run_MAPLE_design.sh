# =============================================================================
# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: GPL-3.0-or-later
# =============================================================================
#!/bin/bash
# =============================================================================
# SCRIPT  : run_MAPLE_design.sh
# PURPOSE : Automated end-to-end probe design pipeline for MAPLE platform
#            (Methylation-Anchor Probe for Low Enrichment)
#
# OVERVIEW:
#   This script orchestrates the MAPLE probe design workflow, including:
#     1. Sequence extension and haplotype-based fragment extraction
#     2. Sliding probe generation across defined length windows
#     3. Thermodynamic filtering (ΔG, GC, Tm constraints)
#     4. Off-target screening via BLASTN (methylated / unmethylated references)
#     5. Consolidation and ranking of candidate probes
#
# INPUTS  :
#   - Reference genome FASTA
#   - Haplotype definition table (TSV)
#
# OUTPUTS :
#   - Optimized hypermethylated and hypomethylated probe sets
#
# USAGE   :
#   bash run_MAPLE_design.sh -r <reference.fa> -i <haplotypes.tsv> [options]
#
# AUTHOR  : Yangjunyi Li
# CREATED : 2023-08-01
# UPDATED : 2025-07-10
#
# NOTE    :
#   This script is designed for reproducible probe generation in MAPLE-TBS
#   workflows and supports both CTOT/CTOB and OT/OB chain designs.
# =============================================================================

set -euo pipefail

# ============ Configuration ============

# ============ Default Values ============
script_path="./"
ref="hs37d5+J02459.1+L09137.2+EBV+HBV+HPV.fasta"
input="${HOME}/CRC_haplotypes.tsv"
prefix="MAPLE_CRC"
extend=45
chain="CTOTCTOB"
min_len=25
max_len=65

# ============ Help Message Function ============
usage() {
    echo ""
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  -s  Path to MAPLE script directory (default: ./)"
    echo "  -r  Reference FASTA file (default: hs37d5+J02459.1+...fa)"
    echo "  -i  Input haplotype table (default: \$HOME/CRC_haplotypes.tsv)"
    echo "  -p  Output prefix (default: MAPLE_CRC)"
    echo "  -e  Sequence extension size (default: 45)"
    echo "  -c  Probe chain type: OTOB or CTOTCTOB (default: CTOTCTOB)"
    echo "  -l  Minimum probe length (default: 25)"
    echo "  -L  Maximum probe length (default: 65)"
    echo "  -h  Show this help message and exit"
    echo ""
    exit 1
}

# ============ Parse Arguments ============
while getopts "s:r:i:p:e:c:l:L:h" opt; do
    case ${opt} in
        s) script_path="$OPTARG" ;;
        r) ref="$OPTARG" ;;
        i) input="$OPTARG" ;;
        p) prefix="$OPTARG" ;;
        e) extend="$OPTARG" ;;
        c) chain="$OPTARG" ;;
        l) min_len="$OPTARG" ;;
        L) max_len="$OPTARG" ;;
        h) usage ;;
        *) usage ;;
    esac
done

echo ">>> Configuration:"
echo "Script Path  : $script_path"
echo "Reference    : $ref"
echo "Input File   : $input"
echo "Prefix       : $prefix"
echo "Extension    : $extend"
echo "Chain        : $chain"
echo "Min Length   : $min_len"
echo "Max Length   : $max_len"
echo ""

# ============ Step 1: Sequence Generator ============

echo ">> [1/7] Running MAPLE_sequence_generator"
mkdir -p generator
cd generator

python3 ${script_path}/MAPLE_sequence_generator.py \
    --data "${input}" \
    --reference "${ref}" \
    --extend "${extend}" \
    --output_prefix "${prefix}" \
    > sequence_generator.log

mkdir -p hyper hypo
mv *hyper*txt hyper/
mv *hypo*txt hypo/

# ============ Step 2: Sliding Probes ============

echo ">> [2/7] Sliding probes..."
conda activate short_probe

if [[ "$chain" == "OTOB" ]]; then
    target="OX"
    mkdir -p hyper/OX hypo/OX
    cat hyper/${prefix}_hyper_OT_4design.txt hyper/${prefix}_hyper_OB_4design.txt | seqkit tab2fx > hyper/OX/${prefix}_hyper_OX_4design.fa
    cat hypo/${prefix}_hypo_OT_4design.txt hypo/${prefix}_hypo_OB_4design.txt | seqkit tab2fx > hypo/OX/${prefix}_hypo_OX_4design.fa
elif [[ "$chain" == "CTOTCTOB" ]]; then
    target="CTOX"
    mkdir -p hyper/CTOX hypo/CTOX
    cat hyper/${prefix}_hyper_CTOT_4design.txt hyper/${prefix}_hyper_CTOB_4design.txt | seqkit tab2fx > hyper/CTOX/${prefix}_hyper_CTOX_4design.fa
    cat hypo/${prefix}_hypo_CTOT_4design.txt hypo/${prefix}_hypo_CTOB_4design.txt | seqkit tab2fx > hypo/CTOX/${prefix}_hypo_CTOX_4design.fa
else
    echo "ERROR: Unknown chain type '$chain'. Expected OTOB or CTOTCTOB."
    exit 1
fi

for step in $(seq ${min_len} ${max_len}); do
    echo ">> Sliding length: $step"
    seqkit sliding -s 1 -W ${step} hyper/${target}/${prefix}_hyper_${target}_4design.fa | seqkit fx2tab -l -Q > hyper/${target}/${prefix}_hyper_${target}_${step}.txt
    seqkit sliding -s 1 -W ${step} hypo/${target}/${prefix}_hypo_${target}_4design.fa | seqkit fx2tab -l -Q > hypo/${target}/${prefix}_hypo_${target}_${step}.txt
done

# ============ Step 3: Thermodynamic Filtering ============

echo ">> [3/7] Thermodynamic filtering"
mkdir -p ../thermo/hyper/${target} ../thermo/hypo/${target}
ln -sf ${PWD}/hyper/${target}/*txt ../thermo/hyper/${target}/
ln -sf ${PWD}/hypo/${target}/*txt ../thermo/hypo/${target}/
cd ../thermo/hypo/${target}

for step in $(seq ${min_len} ${max_len}); do
    echo ">> Thermo filtering hypo $step"
    python3 ${script_path}/MAPLE_probe_thermodynamics.py \
        --deltaG -21 -18 \
        --gc 0.1 0.9 \
        --probe_length ${step} \
        --prefix ${prefix}_hypo_${target} \
        --tm 45 \
        --salinity 50 \
        --dnac1 250 \
        --dnac2 250 \
        --temp 55 \
        --dntp 0 \
        --output ${prefix}_hypo_${target}
done

cd ../../hyper/${target}
for step in $(seq ${min_len} ${max_len}); do
    echo ">> Thermo filtering hyper $step"
    python3 ${script_path}/MAPLE_probe_thermodynamics.py \
        --deltaG -21 -18 \
        --gc 0.1 0.9 \
        --probe_length ${step} \
        --prefix ${prefix}_hyper_${target} \
        --tm 45 \
        --salinity 50 \
        --dnac1 250 \
        --dnac2 250 \
        --temp 55 \
        --dntp 0 \
        --output ${prefix}_hyper_${target}
done

# ============ Step 4: BLASTN ============

echo ">> [4/7] Running BLASTN"
cd ../../../
mkdir -p blastn/hyper/methyl blastn/hypo/unmethyl

ln -sf thermo/hyper/${target}/*4blastn.txt blastn/hyper/methyl/
ln -sf thermo/hypo/${target}/*4blastn.txt blastn/hypo/unmethyl/

cd blastn/hyper/methyl/
for table in *sub*txt; do
    python3 ${script_path}/MAPLE_parallel_blastn.py \
        --reference ${script_path}/reference/hg19_J024591_L091372_EBV_HBV_HPV_rDNA.CT_GA_combined_conversion_methylated.fa \
        --sub_table "$table"
done

cd ../../hypo/unmethyl/
for table in *sub*txt; do
    python3 ${script_path}/MAPLE_parallel_blastn.py \
        --reference ${script_path}/reference/hg19_J024591_L091372_EBV_HBV_HPV_rDNA.CT_GA_combined_conversion_unmethylated.fa \
        --sub_table "$table"
done

# ============ Step 5: Merge BLAST results ============

echo ">> [5/7] Merging BLAST results"
cd ../../../
mkdir -p filter
ln -sf blastn/hyp*/*/*blastn_res.txt filter/

cd filter
ls *hyper*blastn_res.txt > hyper_blast_res.list
ls *hypo*blastn_res.txt > hypo_blast_res.list
ls *blastn_res.txt > blast_res.list

head -n 1 $(head -n 1 blast_res.list) > ${prefix}_blast_res.txt
head -n 1 $(head -n 1 hyper_blast_res.list) > ${prefix}_hyper_blast_res.txt
head -n 1 $(head -n 1 hypo_blast_res.list) > ${prefix}_hypo_blast_res.txt

cat blast_res.list | xargs -I{} tail -n +2 {} >> ${prefix}_blast_res.txt
cat hyper_blast_res.list | xargs -I{} tail -n +2 {} >> ${prefix}_hyper_blast_res.txt
cat hypo_blast_res.list | xargs -I{} tail -n +2 {} >> ${prefix}_hypo_blast_res.txt

# ============ Step 6: Select Best Probes ============

echo ">> [6/7] Selecting best hyper probes"
python3 ${script_path}/MAPLE_select_best_probe.py \
    --data ${PWD}/${prefix}_hyper_blast_res.txt \
    --output ${prefix}_hyper

echo ">> [7/7] Selecting best hypo probes"
python3 ${script_path}/MAPLE_select_best_probe.py \
    --data ${PWD}/${prefix}_hypo_blast_res.txt \
    --output ${prefix}_hypo

echo ">>> All steps completed successfully."
