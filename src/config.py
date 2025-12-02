"""
Central Configuration for My Netflix Prize Pipeline.

This module contains all configuration parameters and file paths.
I define data locations and model hyperparameters here for centralized management.
"""

import os
from pathlib import Path

# --- Paths ---
# We'll assume this config file is in src/, so project root is one level up.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Raw data (where the text files live)
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

# Processed data (where we dump the parquet files)
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

# Models (where we save the pickle files)
MODELS_DIR = PROJECT_ROOT / "models" / "saved_models"  # Organized them a bit

# Reports (plots, submissions)
REPORTS_DIR = PROJECT_ROOT / "reports"
PLOTS_DIR = REPORTS_DIR / "plots"
SUBMISSION_DIR = REPORTS_DIR / "submissions"

# Ensure these exist
for d in [PROCESSED_DATA_DIR, MODELS_DIR, PLOTS_DIR, SUBMISSION_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# --- Files ---
MOVIES_FILE = PROCESSED_DATA_DIR / "movies.parquet"
RATINGS_FILE = PROCESSED_DATA_DIR / "ratings.parquet"
TRAIN_FILE = PROCESSED_DATA_DIR / "train.parquet"
VAL_FILE = PROCESSED_DATA_DIR / "validation.parquet"
QUALIFYING_FILE = PROCESSED_DATA_DIR / "qualifying.parquet"
PROBE_FILE = PROCESSED_DATA_DIR / "probe.parquet"

# Feature files
USER_FEATURES_FILE = PROCESSED_DATA_DIR / "user_features.parquet"
ITEM_FEATURES_FILE = PROCESSED_DATA_DIR / "item_features.parquet"

# --- Model Params ---
# Random seed for reproducibility (where possible)
SEED = 42

# Sampling
# If the dataset is massive, we sample this many rows for training to save RAM.
MAX_TRAIN_ROWS = 5_000_000 

# Surprise Models
BASELINE_PARAMS = {
    'method': 'als',
    'n_epochs': 5,
    'reg_u': 12,
    'reg_i': 5
}

SVD_PARAMS = {
    'n_epochs': 20,
    'lr_all': 0.005,
    'reg_all': 0.02
}

# XGBoost
XGB_PARAMS = {
    'objective': 'reg:squarederror',
    'n_estimators': 100,
    'learning_rate': 0.1,
    'max_depth': 6,
    'n_jobs': -1,
    'random_state': SEED
}
