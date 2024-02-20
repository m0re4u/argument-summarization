#! /bin/bash

set -e

mkdir -p results/lower_limit_kpg

# KPA
limits=(0.05 0.07 0.14 0.21 0.29 0.36 0.43 0.5 0.57 0.64 0.71 0.79 0.86 0.93 1.0)
for i in "${limits[@]}";
do 
    python3 paper_tables/evaluate_kpg.py  --focus_dataset kpa2021 --kp_cutoff $i 
    python3 paper_tables/evaluate_kpg.py  --focus_dataset hyena --kp_cutoff $i 
    python3 paper_tables/evaluate_kpg.py  --focus_dataset perspectrum --kp_cutoff $i
done
