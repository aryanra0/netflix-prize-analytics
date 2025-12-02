#!/bin/bash
set -e

# Netflix Prize Pipeline Runner
# This script executes the full pipeline: from raw data processing to the interactive dashboard.

echo "1. Setting up environment..."
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
    .venv/bin/pip install -r requirements.txt
fi

PYTHON=".venv/bin/python3"

# Ensure src is in python path so config works
export PYTHONPATH=$PYTHONPATH:.

echo "2. Parsing Data..."
if [ -f "data/processed/ratings.parquet" ]; then
    echo "Ratings data already exists. Skipping parsing (delete it if you want to re-run)."
else
    $PYTHON src/parse.py
fi

echo "3. Splitting Data (PySpark)..."
$PYTHON src/data_split.py

echo "4. Feature Engineering (PySpark)..."
$PYTHON src/featurize.py

echo "5. Training Models..."
# Training uses Pandas/Scikit-Surprise (in-memory)
$PYTHON src/train.py
$PYTHON src/train_features.py

echo "6. Evaluating Models..."
$PYTHON src/evaluate.py

echo "7. Generating Predictions..."
# We default to the SVD model for the final submission
$PYTHON src/predict.py --model models/saved_models/svd_model.pkl

echo "Pipeline Complete!"
echo "Starting Dashboard..."
$PYTHON app.py
