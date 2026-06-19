# AI-Powered Predictive Caching System

This repository contains the data preprocessing pipeline for an AI-Powered Predictive Caching System using Long Short-Term Memory (LSTM) networks. The system predicts which Wikipedia articles will be the most viewed in the upcoming hour based on historical access patterns.

## Data Preprocessing Pipeline

The pipeline transforms raw, compressed Wikipedia pageview logs (billions of records) into structured, mathematical tensors ready for deep learning models. 

It consists of 4 main steps:

### 1. Raw Data Inspection
- **Script:** `python_script_pageview.py`
- **Role:** Reads and inspects massive raw `.gz` pageview dumps directly in the console without extracting them. Useful for debugging and verifying raw data integrity.

### 2. Filtering & Cleaning
- **Script:** `filter_cleaning.py`
- **Role:** Processes the raw hourly `.gz` files. It isolates English Wikipedia traffic (`en` and `en.m`), merges desktop and mobile views, and removes system/noise pages (e.g., `Special:`, `Category:`). Outputs lightweight `..._clean.csv` files.

### 3. Timeseries Matrix Construction
- **Script:** `build_timeseries.py`
- **Role:** Aggregates all hourly cleaned CSVs into a single 2D pivot matrix (Articles $\times$ Hours). It retains only the top 5,000 most popular articles to optimize training efficiency, outputting `timeseries_matrix.csv`.

### 4. LSTM Tensor Preparation (Sliding Window)
- **Script:** `prepare_lstm.py`
- **Role:** Applies a logarithmic transformation (`log1p`) to reduce data skewness. It then constructs sliding windows (e.g., using the past 24 hours to predict the top 100 articles in the next hour). The data is split temporally into Train/Validation/Test sets and saved as a compressed `lstm_data.npz` archive.

## How to use

The final output is a compressed numpy archive (`processed/lstm_data.npz`) containing the $X$ (inputs) and $y$ (labels). 

```python
import numpy as np

# Load the ready-to-train data
data = np.load('processed/lstm_data.npz')
X_train, y_train = data['X_train'], data['y_train']
X_val, y_val     = data['X_val'], data['y_val']
X_test, y_test   = data['X_test'], data['y_test']
```
