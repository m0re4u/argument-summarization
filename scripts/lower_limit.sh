#! /bin/bash


# Max comments per KP
# HyEnA: 9
# KPA: 52
# Perspectrum: 13

mkdir -p results/lower_limit

# KPA
limits=(1 2 3 4 5 6 8 9 12 14 18 22 27 34 42 52)
for i in "${limits[@]}";
do 
    python3 paper_tables/evaluate_kpm.py  --model_dir models/roberta-large-contrastive --focus_dataset kpa2021 --kp_comment_cutoff $i >> results/lower_limit/KPA2021.txt
done

# Hyena
limits=(1 2 3 4 5 6 7 8 9 10)
for i in "${limits[@]}";
do 
    python3 paper_tables/evaluate_kpm.py  --model_dir models/roberta-large-contrastive --focus_dataset hyena --kp_comment_cutoff $i >> results/lower_limit/HyEnA.txt
done

# Perspectrum
limits=(1 2 3 4 5 6 7 8 9 10 11 12 13)
for i in "${limits[@]}";
do 
    python3 paper_tables/evaluate_kpm.py  --model_dir models/roberta-large-contrastive --focus_dataset perspectrum --kp_comment_cutoff $i >> results/lower_limit/Perspectrum.txt
done
