#!/usr/bin/env bash
# Reproduce every number in the paper (CPU, ~1.5-2 h on 2 cores).
set -euo pipefail
python scripts/prepare_data.py --data data/twitter_human_bots_clean.parquet
python -m pytest -q tests
python scripts/run_synthetic.py
python scripts/run_experiments.py --group select main ablation density --threads 2
python scripts/analyze.py
(cd paper && latexmk -pdf -interaction=nonstopmode main.tex)
