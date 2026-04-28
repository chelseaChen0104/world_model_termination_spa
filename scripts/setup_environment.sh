#!/bin/bash
# Setup script for World Model Termination SPA

set -e

echo "============================================================"
echo "Setting up World Model Termination SPA"
echo "============================================================"

# Create virtual environment
echo "[1/4] Creating virtual environment..."
python3 -m venv .venv
source .venv/bin/activate

# Upgrade pip
echo "[2/4] Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "[3/4] Installing dependencies..."
pip install -r requirements.txt

# Install the package in development mode
echo "[4/4] Installing package..."
pip install -e .

echo ""
echo "============================================================"
echo "Setup complete!"
echo "============================================================"
echo ""
echo "To activate the environment:"
echo "  source .venv/bin/activate"
echo ""
echo "To generate SPA data:"
echo "  python scripts/generate_spa_data.py --help"
echo ""
echo "To download official SPA data:"
echo "  python scripts/download_spa_data.py --help"
