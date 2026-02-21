#!/bin/bash
set -euo pipefail

echo "=== ShieldLM Data Download ==="

mkdir -p data/raw

if [ -d "data/raw/InjecAgent" ]; then
    echo "InjecAgent already cloned at data/raw/InjecAgent"
else
    echo "Cloning InjecAgent..."
    git clone https://github.com/uiuc-kang-lab/InjecAgent.git data/raw/InjecAgent
fi

echo "=== Done ==="
echo "Next: python curate_dataset.py --output ./data/unified --injecagent ./data/raw/InjecAgent --benign-samples 8000"
