"""
Model Evaluator.

This script evaluates model performance on the validation set.
It calculates RMSE (Root Mean Squared Error) to quantify prediction accuracy.
It also generates visualization plots to analyze the error distribution.
"""

import pandas as pd
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from surprise import accuracy
from sklearn.metrics import mean_squared_error
from src import config

def evaluate_models():
    val_path = config.VAL_FILE
    models_dir = config.MODELS_DIR
    output_dir = config.PLOTS_DIR
    
    print(f"Loading validation data from {val_path}...")
    df = pd.read_parquet(val_path)
    
    # --- Prepare Data for XGBoost ---
    # XGBoost needs features, not just user_id/movie_id
    print("Loading features for XGBoost evaluation...")
    user_features = pd.read_parquet(config.USER_FEATURES_FILE)
    item_features = pd.read_parquet(config.ITEM_FEATURES_FILE)
    
    df_xgb = df.copy()
    df_xgb = df_xgb.merge(user_features, on='user_id', how='left')
    df_xgb = df_xgb.merge(item_features, on='movie_id', how='left')
    df_xgb = df_xgb.fillna(0)
    
    # Recalculate temporal features for validation set
    df_xgb['date'] = pd.to_datetime(df_xgb['date'])
    df_xgb['year'] = df_xgb['date'].dt.year
    df_xgb['month'] = df_xgb['date'].dt.month
    min_date = df_xgb['date'].min()
    df_xgb['days_since_first'] = (df_xgb['date'] - min_date).dt.days
    
    feature_cols = [
        'user_mean', 'user_count', 'user_std', 
        'movie_mean', 'movie_count', 'movie_std',
        'year', 'month', 'days_since_first'
    ]
    
    # --- Load Models ---
    models = {}
    if not os.path.exists(models_dir):
        print(f"No models found in {models_dir}. Did you run training?")
        return

    for filename in os.listdir(models_dir):
        if filename.endswith('.pkl'):
            model_name = filename.replace('.pkl', '')
            print(f"Loading {model_name}...")
            with open(models_dir / filename, 'rb') as f:
                models[model_name] = pickle.load(f)
    
    results = []
    
    # --- Evaluate Each Model ---
    for name, algo in models.items():
        print(f"Evaluating {name}...")
        
        if hasattr(algo, 'test'):
            # It's a Surprise model (SVD, Baseline)
            # Surprise expects a list of (user, item, rating) tuples
            testset = list(zip(df['user_id'], df['movie_id'], df['rating']))
            preds = algo.test(testset)
            
            rmse = accuracy.rmse(preds, verbose=True)
            y_true = [p.r_ui for p in preds]
            y_est = [p.est for p in preds]
            
        else:
            # It's likely XGBoost (or Scikit-Learn style)
            X_val = df_xgb[feature_cols]
            y_true = df_xgb['rating'].values
            y_est = algo.predict(X_val)
            
            rmse = np.sqrt(mean_squared_error(y_true, y_est))
            print(f"RMSE: {rmse:.4f}")

        results.append({'model': name, 'rmse': rmse})
        
        # --- Plot Residuals ---
        # Residual = True - Predicted. 
        # Ideally, this should be a nice bell curve centered at 0.
        residuals = np.array(y_true) - np.array(y_est)
        plt.figure(figsize=(10, 6))
        sns.histplot(residuals, bins=50, kde=True)
        plt.title(f'Residuals Distribution - {name}')
        plt.xlabel('Residual (True - Predicted)')
        
        plot_path = output_dir / f'residuals_{name}.png'
        plt.savefig(plot_path)
        plt.close()
        
    # Save summary results
    results_df = pd.DataFrame(results)
    print("\nFinal Results:")
    print(results_df)
    results_df.to_csv(output_dir / 'evaluation_results.csv', index=False)

if __name__ == "__main__":
    evaluate_models()
