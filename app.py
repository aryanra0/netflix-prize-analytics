from flask import Flask, render_template, jsonify, request
import pandas as pd
import numpy as np
import os
import pickle
import random
from datetime import datetime, timedelta

app = Flask(__name__)

# --- Dummy Data Generation (Simulation) ---
def generate_dummy_data():
    """Generates synthetic data for visualization if real files are missing."""
    print("Generating dummy data for dashboard...")
    
    # 1. Model Comparison
    models = ['Baseline (ALS)', 'SVD', 'XGBoost', 'SVD++']
    rmse_scores = [1.02, 0.98, 0.96, 0.95]
    model_metrics = [{'model': m, 'rmse': r} for m, r in zip(models, rmse_scores)]
    
    # 2. Prediction Accuracy (Scatter)
    # Simulate 500 predictions
    actuals = np.random.randint(1, 6, 500)
    # Add some noise to create predictions
    preds = actuals + np.random.normal(0, 0.5, 500)
    preds = np.clip(preds, 1, 5)
    scatter_data = [{'actual': int(a), 'predicted': round(float(p), 2)} for a, p in zip(actuals, preds)]
    
    # 3. Error Distribution
    # Residuals for each rating class
    residuals = {}
    for r in range(1, 6):
        # Simulate errors: normally distributed around 0
        res = np.random.normal(0, 0.8, 100).tolist()
        residuals[r] = res
        
    # 4. Temporal Trends
    # Monthly average ratings over 3 years
    dates = pd.date_range(start='2020-01-01', periods=36, freq='M')
    trends = []
    base_rating = 3.5
    for d in dates:
        # Add some seasonality/trend
        rating = base_rating + (np.sin(d.month) * 0.2) + np.random.normal(0, 0.1)
        trends.append({'date': d.strftime('%Y-%m'), 'rating': round(rating, 2)})
        
    # 5. Feature Importance (XGBoost)
    features = {
        'User Mean Rating': 0.35,
        'Item Mean Rating': 0.25,
        'Days Since First Rating': 0.15,
        'Rating Count (User)': 0.10,
        'Rating Count (Item)': 0.08,
        'Year': 0.05,
        'Month': 0.02
    }
    feature_importance = [{'feature': k, 'importance': v} for k, v in features.items()]
    
    return {
        'model_metrics': model_metrics,
        'scatter_data': scatter_data,
        'residuals': residuals,
        'trends': trends,
        'feature_importance': feature_importance
    }

# Cache the data in memory
DASHBOARD_DATA = generate_dummy_data()

# --- Routes ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/metrics')
def get_metrics():
    """Returns model comparison metrics."""
    return jsonify(DASHBOARD_DATA['model_metrics'])

@app.route('/api/accuracy')
def get_accuracy():
    """Returns data for Actual vs Predicted scatter plot."""
    # Optional: Filter by 'set' (validation vs test) if we had it
    dataset = request.args.get('dataset', 'validation')
    # In a real app, we'd load different data here. 
    # For now, just return the same dummy data or slightly modified
    data = DASHBOARD_DATA['scatter_data']
    if dataset == 'test':
        # Simulate slightly different data for test set
        data = [{'actual': d['actual'], 'predicted': min(5, max(1, d['predicted'] + 0.1))} for d in data]
    return jsonify(data)

@app.route('/api/errors')
def get_errors():
    """Returns residuals distribution."""
    return jsonify(DASHBOARD_DATA['residuals'])

@app.route('/api/trends')
def get_trends():
    """Returns temporal trends."""
    return jsonify(DASHBOARD_DATA['trends'])

@app.route('/api/features')
def get_features():
    """Returns feature importance."""
    return jsonify(DASHBOARD_DATA['feature_importance'])

@app.route('/api/predict', methods=['POST'])
def predict():
    """Live Prediction Sandbox Endpoint."""
    try:
        data = request.json
        user_id = int(data.get('user_id', 0))
        movie_id = int(data.get('movie_id', 0))
        model_type = data.get('model_type', 'svd')
        
        # Simulate a prediction since we might not have the loaded models in memory
        # In a real app, you would load the pickle files and call .predict()
        
        # Deterministic "random" prediction based on inputs so it feels consistent
        random.seed(user_id + movie_id)
        base_pred = 3.5
        noise = random.uniform(-1.0, 1.5)
        prediction = max(1.0, min(5.0, base_pred + noise))
        
        return jsonify({
            'user_id': user_id,
            'movie_id': movie_id,
            'model': model_type,
            'prediction': round(prediction, 2),
            'status': 'success'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
