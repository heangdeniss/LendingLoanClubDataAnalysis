# Data Source

https://www.kaggle.com/datasets/wordsforthewise/lending-club

# LendingClub Loan Default Prediction

This repository contains exploratory data analysis, preprocessing, and model development for predicting LendingClub loan default (loan status) from a concatenated LendingClub dataset. The work is organized as Jupyter notebooks and uses DuckDB for fast ingestion and Parquet storage.

## Contents

- `lendingClubDefualtPrediction.ipynb` — Main end-to-end notebook: data loading, cleaning, feature engineering, model training (ANN, XGBoost, Random Forest), evaluation, and visualizations.
- `concat_data.ipynb` — Notebook used for concatenating and preparing raw CSV data.
- `readme.md` — This file.

## Quickstart

1. Open the notebooks in Jupyter, VS Code, or another notebook runner.
2. Install Python dependencies (recommended in a virtual environment):

```bash
pip install -r requirements.txt
# or, if you don't have requirements.txt, run:
pip install pandas numpy scipy scikit-learn xgboost tensorflow matplotlib seaborn duckdb holoviews hvplot
```

Notes:
- GPU-enabled training for XGBoost requires an XGBoost build with GPU support; set `tree_method='gpu_hist'` and `predictor='gpu_predictor'` when available.
- scikit-learn's `RandomForestClassifier` does not support GPU. For GPU random-forest-style training, consider RAPIDS cuML (requires CUDA and conda installation).

## Data

- The workflow reads raw CSV(s) and converts them to Parquet using DuckDB for faster loading. The Parquet file used is `concat_accepted_rejected.parquet` (created by the notebooks).
- Data cleaning steps include NA removal, outlier filtering, feature dropping, and dummy encoding. See `lendingClubDefualtPrediction.ipynb` for details.

## Preprocessing Steps (high-level)

- Load Parquet dataset via DuckDB / pandas
- Remove or impute missing values
- Drop redundant features (e.g., `emp_title`, `title`) and avoid leakage (drop `issue_d`)
- Convert dates (`earliest_cr_line`) and extract year
- Create dummy variables for categorical fields (e.g., `sub_grade`, `purpose`, `home_ownership`, `zip_code`)
- Scale numeric features and filter extreme outliers

## Models & Evaluation

- ANN (TensorFlow Keras): multi-layer network with learning-curve visualization and AUC monitoring.
- XGBoost: randomized hyperparameter search (`RandomizedSearchCV`) and evaluation with ROC AUC; supports GPU via `gpu_hist` when available.
- Random Forest: CPU tuning via `RandomizedSearchCV`; for GPU alternative see RAPIDS cuML example in the notebook discussion.

Evaluation artifacts include confusion matrices, ROC curves, and classification reports. Results are collected in a `scores_dict` variable inside the notebook.
