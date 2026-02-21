#!/bin/bash
set -euo pipefail

echo "=== ShieldLM Full Pipeline ==="

echo "--- Step 1: Download data ---"
bash scripts/download_data.sh

echo "--- Step 2: Curate dataset ---"
python curate_dataset.py \
    --output ./data/unified \
    --injecagent ./data/raw/InjecAgent \
    --benign-samples 8000

echo "--- Step 3: Train DeBERTa-v3-base ---"
accelerate launch --num_processes=2 -m shieldlm.train \
    --config configs/deberta_base.yaml

echo "--- Step 4: Evaluate ---"
python -m shieldlm.evaluate \
    --model models/deberta-v3-base-shieldlm \
    --test-data data/unified/test.parquet \
    --output results/eval_report.json \
    --baselines

echo "=== Pipeline complete ==="
