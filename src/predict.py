"""
Prediction Generator.

This script generates the final predictions. I take my trained model and predict ratings for the "Qualifying" set.
The output format is specific to the Netflix Prize (MovieID:, then UserID,Rating).
"""

import pandas as pd
import pickle
import xgboost as xgb
import argparse
from src import config

def load_model(model_path):
    with open(model_path, 'rb') as f:
        return pickle.load(f)

def predict_qualifying(model_path):
    # We'll default to SVD if not specified, but usually this is passed in.
    output_path = config.SUBMISSION_DIR / 'submission.txt'
    
    print(f"Loading model from {model_path}...")
    model = load_model(model_path)
    
    print(f"Loading qualifying data from {config.QUALIFYING_FILE}...")
    df = pd.read_parquet(config.QUALIFYING_FILE)
    
    # Figure out what kind of model we're dealing with
    is_xgboost = isinstance(model, xgb.XGBRegressor)
    is_surprise = not is_xgboost
    
    predictions = []
    
    if is_surprise:
        print("Predicting with Surprise model (this might take a while loop-de-loop)...")
        # Surprise's predict() is single-row, so we loop. 
        # Not the fastest, but it works.
        for _, row in df.iterrows():
            pred = model.predict(row['user_id'], row['movie_id'])
            predictions.append(pred.est)
            
    elif is_xgboost:
        print("Predicting with XGBoost model...")
        
        # Need to reconstruct features for the test set
        print("Loading features...")
        user_features = pd.read_parquet(config.USER_FEATURES_FILE)
        item_features = pd.read_parquet(config.ITEM_FEATURES_FILE)
        
        df = df.merge(user_features, on='user_id', how='left')
        df = df.merge(item_features, on='movie_id', how='left')
        df = df.fillna(0)
        
        # Time features
        df['date'] = pd.to_datetime(df['date'])
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        min_date = df['date'].min()
        df['days_since_first'] = (df['date'] - min_date).dt.days
        
        feature_cols = [
            'user_mean', 'user_count', 'user_std', 
            'movie_mean', 'movie_count', 'movie_std',
            'year', 'month', 'days_since_first'
        ]
        
        X = df[feature_cols]
        predictions = model.predict(X)
        
    df['rating'] = predictions
    
    # Format for submission:
    # MovieID:
    # UserID,Rating
    # ...
    print(f"Writing submission to {output_path}...")
    
    # We sort by MovieID then UserID to match the format requirement
    df_sorted = df.sort_values(['movie_id', 'user_id'])
    
    with open(output_path, 'w') as f:
        current_movie_id = None
        for _, row in df_sorted.iterrows():
            movie_id = int(row['movie_id'])
            user_id = int(row['user_id'])
            rating = row['rating']
            
            if movie_id != current_movie_id:
                f.write(f"{movie_id}:\n")
                current_movie_id = movie_id
            
            f.write(f"{user_id},{rating:.3f}\n")
            
    print("Done! Submission file is ready.")

def main():
    parser = argparse.ArgumentParser(description="Generate Predictions")
    parser.add_argument('--model', type=str, required=True, help='Path to model pickle')
    args = parser.parse_args()
    
    predict_qualifying(args.model)

if __name__ == "__main__":
    main()
