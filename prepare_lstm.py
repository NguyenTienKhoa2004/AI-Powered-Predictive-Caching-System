"""
prepare_lstm.py
===============
Chuan bi data tu timeseries_matrix.csv de train LSTM predictive caching.

Pipeline:
  1. Load timeseries_matrix.csv
  2. Filter: bo bai co bat ky gio nao = 0 (breaking news, bai moi)
  3. Normalize: log1p(views) de giam skewness
  4. Sliding window: tao sequences (X, y) cho LSTM
       X shape: (n_samples, window_size, 1)  <- input sequence
       y shape: (n_samples, top_k)           <- label: top-K bai duoc xem nhieu nhat
  5. Train/Val/Test split (theo thoi gian, khong random)
  6. Luu ra .npz

Output:
  processed/lstm_data.npz   <- load bang np.load()

Cach dung sau khi co file:
  data   = np.load('lstm_data.npz')
  X_train, y_train = data['X_train'], data['y_train']
  X_val,   y_val   = data['X_val'],   data['y_val']
  X_test,  y_test  = data['X_test'],  data['y_test']
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path

# =============================================================================
# CONFIG
# =============================================================================
MATRIX_FILE = Path(r"C:\Users\as\OneDrive\Desktop\Thesis\processed\timeseries_matrix.csv")
OUTPUT_DIR  = Path(r"C:\Users\as\OneDrive\Desktop\Thesis\processed")

WINDOW_SIZE = 24      

TOP_K = 100           

TRAIN_RATIO = 0.67    
VAL_RATIO   = 0.16    


# =============================================================================
# STEP 1: Load
# =============================================================================
def load_matrix(path: Path) -> pd.DataFrame:
    print(f"Loading {path.name}...")
    df = pd.read_csv(path, index_col=0)
    # Chuyen ten cot thanh Timestamp
    df.columns = pd.to_datetime(df.columns)
    df = df.sort_index(axis=1)   # dam bao thu tu thoi gian
    print(f"  Shape: {df.shape[0]:,} articles x {df.shape[1]} hours")
    return df


# =============================================================================
# STEP 2: Filter 
# =============================================================================
def filter_complete(df: pd.DataFrame) -> pd.DataFrame:
    n_before = len(df)
    df = df[(df == 0).sum(axis=1) == 0]   # chi giu bai xuat hien du 72/72 gio
    n_after = len(df)
    print(f"Filter zeros: {n_before:,} -> {n_after:,} articles "
          f"(removed {n_before - n_after} breaking-news / incomplete)")
    return df


# =============================================================================
# STEP 3: Normalize
# =============================================================================
def normalize(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    Ap dung log1p de giam skewness (views range tu 1 -> 261,621).
    Tra ve:
      arr_norm  : (n_articles, n_hours) float32, da normalize
      arr_raw   : (n_articles, n_hours) int32, views goc de tinh label
    """
    arr_raw  = df.values.astype(np.int32)
    arr_norm = np.log1p(arr_raw).astype(np.float32)
    print(f"Normalize log1p: raw range [{arr_raw.min()}, {arr_raw.max()}] "
          f"-> norm range [{arr_norm.min():.2f}, {arr_norm.max():.2f}]")
    return arr_norm, arr_raw


# =============================================================================
# STEP 4: Sliding window
# =============================================================================
def make_sequences(arr_norm: np.ndarray,
                   arr_raw:  np.ndarray,
                   window:   int,
                   top_k:    int) -> tuple[np.ndarray, np.ndarray]:
    """
    Voi moi time step t (tu window den n_hours-1):
      X[t] = arr_norm[:, t-window : t]   shape: (n_articles, window)
             -> transpose + reshape -> (window, n_articles) cho LSTM
      y[t] = one-hot vector: bai nao nam trong top-K views cao nhat o gio t?
             shape: (n_articles,)  -- multi-label binary

    Return:
      X : (n_timesteps, window, n_articles)  float32
      y : (n_timesteps, n_articles)          float32 (0/1 label)
    """
    n_articles, n_hours = arr_norm.shape
    n_steps = n_hours - window

    print(f"Creating sliding windows: "
          f"window={window}h, steps={n_steps}, articles={n_articles:,}, top_k={top_k}")

    X_list, y_list = [], []

    for t in range(window, n_hours):
        x = arr_norm[:, t - window : t]   
        x = x.T                            
        X_list.append(x)

        views_at_t = arr_raw[:, t]
        label = np.zeros(n_articles, dtype=np.float32)
        top_k_idx = np.argpartition(views_at_t, -top_k)[-top_k:]
        label[top_k_idx] = 1.0
        y_list.append(label)

    X = np.stack(X_list, axis=0)   
    y = np.stack(y_list, axis=0)   

    print(f"  X shape: {X.shape}  (timesteps, window, articles)")
    print(f"  y shape: {y.shape}  (timesteps, articles) -- multi-label top-{top_k}")

    return X, y


# =============================================================================
# STEP 5: Train/Val/Test split 
# =============================================================================
def temporal_split(X: np.ndarray, y: np.ndarray,
                   train_r: float, val_r: float
                   ) -> tuple:
    n = len(X)
    i_train = int(n * train_r)
    i_val   = int(n * (train_r + val_r))

    X_train, y_train = X[:i_train],        y[:i_train]
    X_val,   y_val   = X[i_train:i_val],   y[i_train:i_val]
    X_test,  y_test  = X[i_val:],          y[i_val:]

    print(f"Split (temporal):")
    print(f"  Train : {len(X_train):>4} steps  ({len(X_train)/n*100:.0f}%)")
    print(f"  Val   : {len(X_val):>4} steps  ({len(X_val)/n*100:.0f}%)")
    print(f"  Test  : {len(X_test):>4} steps  ({len(X_test)/n*100:.0f}%)")

    return X_train, y_train, X_val, y_val, X_test, y_test


# =============================================================================
# MAIN
# =============================================================================
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_matrix(MATRIX_FILE)

    df = filter_complete(df)

    titles = df.index.tolist()
    title_map_path = OUTPUT_DIR / "lstm_title_index.json"
    with open(title_map_path, 'w', encoding='utf-8') as f:
        json.dump({i: t for i, t in enumerate(titles)}, f, ensure_ascii=False, indent=2)
    print(f"Saved title index -> {title_map_path.name}  ({len(titles):,} articles)")

    arr_norm, arr_raw = normalize(df)

    X, y = make_sequences(arr_norm, arr_raw, window=WINDOW_SIZE, top_k=TOP_K)

    X_train, y_train, X_val, y_val, X_test, y_test = temporal_split(
        X, y, TRAIN_RATIO, VAL_RATIO
    )

    out_path = OUTPUT_DIR / "lstm_data.npz"
    np.savez_compressed(
        out_path,
        X_train=X_train, y_train=y_train,
        X_val=X_val,     y_val=y_val,
        X_test=X_test,   y_test=y_test,
    )

    size_mb = out_path.stat().st_size / 1e6
    print(f"\n[SAVED] {out_path.name}  ({size_mb:.1f} MB)")
    print("\nDone! Load bang:")
    print("  data = np.load('processed/lstm_data.npz')")
    print("  X_train, y_train = data['X_train'], data['y_train']")


if __name__ == "__main__":
    main()
