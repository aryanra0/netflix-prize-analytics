"""
Collaborative Filtering Model Trainer.

This script trains the "classic" recommender models using the Surprise library.
I'm training:
1. BaselineOnly (ALS): A simple model that just looks at user/item biases.
2. SVD: The heavy hitter. Matrix Factorization that finds hidden patterns.
"""

import pandas as pd
import pickle
from surprise import Dataset, Reader, SVD, BaselineOnly
from src import config

def train_models():
    train_path = config.TRAIN_FILE
    models_dir = config.MODELS_DIR
    
    print(f"Loading training data from {train_path}...")
    df = pd.read_parquet(train_path)
    
    # Safety check: If the dataset is massive, we sample it down.
    # Otherwise, we'll blow up the RAM.
    if len(df) > config.MAX_TRAIN_ROWS:
        print(f"Dataset is huge ({len(df)} rows). Sampling {config.MAX_TRAIN_ROWS} rows...")
        df = df.sample(n=config.MAX_TRAIN_ROWS, random_state=config.SEED)
    
    # Surprise needs to know the rating scale (1-5 stars)
    reader = Reader(rating_scale=(1, 5))
    
    print("Converting to Surprise Dataset...")
    data = Dataset.load_from_df(df[['user_id', 'movie_id', 'rating']], reader)
    
    # Build the full trainset (no internal split, we have a separate validation set)
    trainset = data.build_full_trainset()
    
    # --- Model 1: Baseline (ALS) ---
    print("Training Baseline Model (ALS)...")
    algo_bsl = BaselineOnly(bsl_options=config.BASELINE_PARAMS)
    algo_bsl.fit(trainset)
    
    bsl_path = models_dir / 'baseline_model.pkl'
    print(f"Saving Baseline model to {bsl_path}...")
    with open(bsl_path, 'wb') as f:
        pickle.dump(algo_bsl, f)
        
    # --- Model 2: SVD ---
    print("Training SVD Model...")
    algo_svd = SVD(**config.SVD_PARAMS)
    algo_svd.fit(trainset)
    
    svd_path = models_dir / 'svd_model.pkl'
    print(f"Saving SVD model to {svd_path}...")
    with open(svd_path, 'wb') as f:
        pickle.dump(algo_svd, f)
        
    print("Training complete. Models are baked.")

if __name__ == "__main__":
    train_models()
