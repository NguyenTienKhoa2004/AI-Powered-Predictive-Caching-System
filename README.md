# AI-Powered Predictive Caching System

An intelligent caching system powered by LSTM neural networks that predicts which Wikipedia articles will be most viewed in the upcoming hour based on historical access patterns.

## Data Preprocessing Pipeline

The pipeline transforms raw, compressed Wikipedia pageview logs into structured tensors ready for LSTM training in 3 steps:

```
  Raw .gz logs ──► Clean CSVs ──► Timeseries Matrix ──► LSTM Tensors (.npz)
```

---

### Step 1: Filtering & Cleaning (`filter_cleaning.py`)

Reads raw compressed `.gz` pageview dump files from Wikimedia servers and produces clean, filtered CSVs.

**Input:** Compressed `pageviews-YYYYMMDD-HHMMSS.gz` — each line is a raw access log entry:
```
fr  Wikipedia:Accueil_principal  1250  0
en  Albert_Einstein              500   0
en  Special:Search               8900  0      ← system page (noise)
en.m Python_(programming_language) 180 0      ← mobile traffic
de  Hauptseite                   300   0      ← non-English (removed)
```

**Processing:**
- Keep only English Wikipedia traffic (`en` + `en.m`)
- Remove system/noise pages (`Special:`, `Category:`, `User:`, `Talk:`...)
- Remove the main page (`Main_Page`)
- Merge Desktop + Mobile view counts per article

**Output:** `pageviews-..._clean.csv` — one file per hour of clean data:
| title | views |
| :--- | ---: |
| Albert_Einstein | 500 |
| Python_(programming_language) | 180 |
| Machine_learning | 320 |

> 72 raw `.gz` files → 72 `_clean.csv` files

---

### Step 2: Timeseries Matrix (`build_timeseries.py`)

Merges all 72 hourly CSV files into a **single 2D pivot matrix** (articles × hours).

**Processing:**
- **Pass 1:** Sum total views across all 72 hours → select the **Top 5,000** most popular articles
- **Pass 2:** Re-read each file, keep only Top 5,000 → assemble into a pivot table

**Output:** `timeseries_matrix.csv` — a `5,000 articles × 72 hours` matrix:
| title | 01-01 00:00 | 01-01 01:00 | 01-01 02:00 | ... | 01-03 23:00 |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Albert_Einstein | 500 | 480 | 390 | ... | 450 |
| Machine_learning | 320 | 340 | 380 | ... | 500 |
| Python_(programming_...) | 180 | 195 | 150 | ... | 210 |

---

### Step 3: LSTM Tensor Preparation (`prepare_lstm.py`)

Transforms the timeseries matrix into 3D tensors ready for LSTM training.

**Processing:**
1. **Filter:** Remove articles with any hour having 0 views (4,883 articles remaining)
2. **Normalize:** Apply `log1p(views)` — reduces skewness between articles with 1 view vs. 261,621 views
3. **Sliding Window:** Use the **past 24 hours** as input to predict **the next hour**
4. **Labeling:** Top 100 most-viewed articles in the predicted hour → label = 1, rest = 0
5. **Temporal Split:** Train 67% | Val 16% | Test 17% (chronological order, no shuffling)

**Output:** `lstm_data.npz` containing:
```
X_train: (32, 24, 4883)  →  32 samples × 24h history × 4,883 articles
y_train: (32, 4883)      →  32 samples × binary label per article (top-100)
X_val:   (7, 24, 4883)
y_val:   (7, 4883)
X_test:  (9, 24, 4883)
y_test:  (9, 4883)
```

---

## How to Use

```python
import numpy as np

data = np.load('processed/lstm_data.npz')
X_train, y_train = data['X_train'], data['y_train']
X_val, y_val     = data['X_val'], data['y_val']
X_test, y_test   = data['X_test'], data['y_test']
```
