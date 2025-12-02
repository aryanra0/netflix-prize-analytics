# 🍿 Netflix Prize Analytics

**Project Overview:**
This project implements an end-to-end Machine Learning pipeline to predict user ratings, inspired by the Netflix Prize competition. I process the dataset using PySpark and train multiple models (SVD, XGBoost) to optimize prediction accuracy. The results are visualized in a custom, interactive dashboard for analysis.

---

## 🚀 Quick Start

Get this running on your machine in under 2 minutes.

```bash
# 1. Clone the repo
git clone <repo-url>
cd "NTFLX Project"

# 2. Run the full pipeline (Setup -> Process -> Train -> Dashboard)
./run.sh
```

That's it! The script will handle the virtual environment, dependencies, and data processing.

---

## 📊 Visualizations

I built a custom **Interactive Dashboard** to explore the results.

**How to view it:**
The `./run.sh` script launches it automatically at the end. If you want to run *just* the dashboard:

```bash
.venv/bin/python app.py
```

Then open **[http://127.0.0.1:5000](http://127.0.0.1:5000)** in your browser.

**What you'll see:**
*   **Model Comparison**: See which model (SVD vs XGBoost) wins on RMSE.
*   **Live Sandbox**: Enter a User ID and Movie ID to get a real-time rating prediction.
*   **Deep Dives**: Scatter plots of accuracy and error distributions.

---

## 🏗 Architecture

Here is an overview of the project structure and pipeline:

*   **`src/parse.py`**: Converts the clunky raw text files into fast **Parquet** files.
*   **`src/data_split.py`**: Uses **PySpark** to split the data into Training and Validation sets (using the official Probe set).
*   **`src/featurize.py`**: Calculates user stats (mean rating, count) and movie stats using PySpark.
*   **`src/train.py`**: Trains Collaborative Filtering models (SVD, Baseline) using `scikit-surprise`.
*   **`src/train_features.py`**: Trains a Gradient Boosting model (XGBoost) using the engineered features.
*   **`app.py`**: A Flask backend that serves the data to the frontend.

---

## 🧪 Testing

I have unit tests to make sure the data parsing logic isn't broken.

```bash
# Run tests
.venv/bin/pytest tests/
```

---

## 🛠 Troubleshooting

*   **Java Error?** PySpark needs Java 8 or 11. Make sure `JAVA_HOME` is set.
*   **"Killed: 9"?** You ran out of RAM. I automatically sample the data to 5M rows in `src/config.py`, but you can lower `MAX_TRAIN_ROWS` if needed.
*   **Missing Data?** Ensure the raw Netflix Prize files are in `data/raw/`.

---

## 📜 License & Credits

This project is for educational purposes. The dataset is property of Netflix (or whoever owns it now).
This is my personal project. Feel free to fork and break things!
