#!/bin/bash
# ─────────────────────────────────────────────────────────────────
#  run.sh  —  Full pipeline: setup → download → train → visualize
#  Usage:  bash run.sh
# ─────────────────────────────────────────────────────────────────

set -e  # exit on error

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  Week 7 · RNN & Seq2Seq Forecasting Project"
echo "════════════════════════════════════════════════════════════"

# Step 1: Create virtual env (optional — comment out if using conda/system)
echo ""
echo "▶  [1/4] Installing dependencies..."
pip install -r requirements.txt --quiet

# Step 2: Download data
echo ""
echo "▶  [2/4] Downloading dataset..."
python download_data.py

# Step 3: Train all models
echo ""
echo "▶  [3/4] Training all 7 models (this will take ~15-30 min on CPU)..."
python train.py

# Step 4: Generate visualizations
echo ""
echo "▶  [4/4] Generating premium visualizations..."
python visualize.py --dpi 200

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  ✅  Done!  Check the plots/ directory for all figures."
echo "════════════════════════════════════════════════════════════"
echo ""
