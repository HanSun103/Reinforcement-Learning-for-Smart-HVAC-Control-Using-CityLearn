#!/bin/bash
# ==========================================================================
# install.sh — CityLearn HVAC RL Project (macOS / Linux)
# Run from the project root:  bash install.sh
# ==========================================================================

set -e

echo ""
echo "============================================================"
echo " CityLearn HVAC RL — dependency installer (macOS/Linux)"
echo "============================================================"
echo ""

# Step 1: PyTorch CPU-only
echo "[1/5] Installing PyTorch (CPU-only)..."
pip install "torch>=2.0,<2.5" --index-url https://download.pytorch.org/whl/cpu

# Step 2: CityLearn without openstudio
echo ""
echo "[2/5] Installing CityLearn 2.5.0 (skipping openstudio)..."
pip install citylearn==2.5.0 --no-deps

# Step 3: CityLearn minimal deps
echo ""
echo "[3/5] Installing CityLearn minimal dependencies..."
pip install "gymnasium==0.28.1" "scikit-learn==1.2.2" simplejson pyyaml

# Step 4: SB3 pinned — last version for gymnasium 0.28
echo ""
echo "[4/5] Installing Stable Baselines3 2.2.1..."
pip install "stable-baselines3==2.2.1" --no-deps
pip install cloudpickle

# Step 5: Utilities
echo ""
echo "[5/5] Installing matplotlib, pandas, numpy..."
pip install "matplotlib>=3.7.0" "pandas>=2.0.0" "numpy>=1.24.0"

# Verify
echo ""
echo "============================================================"
echo " Verifying..."
echo "============================================================"
python -c "import torch; print('torch    :', torch.__version__)"
python -c "import gymnasium; print('gymnasium:', gymnasium.__version__)"
python -c "import stable_baselines3; print('sb3      :', stable_baselines3.__version__)"
python -c "import citylearn; print('citylearn:', citylearn.__version__)"

echo ""
echo "============================================================"
echo " Done! Run:  python src/quick_test.py"
echo "============================================================"
echo ""
