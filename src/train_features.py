"""
XGBoost Model Trainer.

This script trains the XGBoost model. I combine the pre-computed user/item statistics,
incorporate temporal features (e.g., user tenure), and train the gradient boosting regressor.
"""

import pandas as pd
import pickle
import xgboost as xgb
from src import config

def train_xgboost():
    print(f"Loading training data from {config.TRAIN_FILE}...")
    df = pd.read_parquet(config.TRAIN_FILE)

    # Sampling again to save RAM
    if len(df) > config.MAX_TRAIN_ROWS:
        print(f"Dataset is huge ({len(df)} rows). Sampling {config.MAX_TRAIN_ROWS} rows...")
        df = df.sample(n=config.MAX_TRAIN_ROWS, random_state=config.SEED)
    
    print("Loading pre-computed features...")
    user_features = pd.read_parquet(config.USER_FEATURES_FILE)
    item_features = pd.read_parquet(config.ITEM_FEATURES_FILE)
    
    print("Joining features...")
    # Merge everything together. Left join because we want to keep all training rows.
    df = df.merge(user_features, on='user_id', how='left')
    df = df.merge(item_features, on='movie_id', how='left')
    
    # New users/items might have NaNs, fill with 0
    df = df.fillna(0)
    
    # --- Feature Engineering on the fly ---
    # We calculate these here because they depend on the specific dataset's date range
    print("Computing temporal features...")
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    
    # "Days since first rating" - a proxy for user tenure/platform age
    min_date = df['date'].min()
    df['days_since_first'] = (df['date'] - min_date).dt.days
    
    feature_cols = [
        'user_mean', 'user_count', 'user_std', 
        'movie_mean', 'movie_count', 'movie_std',
        'year', 'month', 'days_since_first'
    ]
    
    X = df[feature_cols]
    y = df['rating']
    
    print("Training XGBoost Model...")
    # Using params from config
    model = xgb.XGBRegressor(**config.XGB_PARAMS)
    model.fit(X, y)
    
    model_path = config.MODELS_DIR / 'xgboost_model.pkl'
    print(f"Saving XGBoost model to {model_path}...")
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
        
    print("Training complete. XGBoost is ready to boost.")

if __name__ == "__main__":
    train_xgboost()
