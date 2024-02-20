#! /bin/bash


# Perspectrum
mkdir -p results/perspectrum_sources/
echo "--- idebate ---" >> results/perspectrum_sources/Perspectrum.txt
python3 paper_tables/evaluate_kpm.py --model_dir models/roberta-large-contrastive  --focus_dataset perspectrum-idebate >> results/perspectrum_sources/Perspectrum.txt
echo "--- procon ---" >> results/perspectrum_sources/Perspectrum.txt
python3 paper_tables/evaluate_kpm.py --model_dir models/roberta-large-contrastive  --focus_dataset perspectrum-procon >> results/perspectrum_sources/Perspectrum.txt
echo "--- debatewise ---" >> results/perspectrum_sources/Perspectrum.txt
python3 paper_tables/evaluate_kpm.py --model_dir models/roberta-large-contrastive  --focus_dataset perspectrum-debatewise >> results/perspectrum_sources/Perspectrum.txt
