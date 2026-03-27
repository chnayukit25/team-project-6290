# src/ - Homomorphic Encryption for Bitcoin Risk Prediction

## Overview

This directory contains the core Python modules for the CS6290 Team Project **"HEBRABT"** (Homomorphic Encryption for Bitcoin Risk Prediction). The project implements three distinct risk‑prediction tasks (Direction A: price crash, Direction B: volatility, Direction C: on‑chain anomaly) using logistic regression models that can be evaluated **entirely in encrypted form** via the CKKS homomorphic‑encryption scheme.

All code is organized into three sub‑packages:

*   **`data/`** – loading and preprocessing of the Bitcoin dataset.
*   **`model/`** – label generation, model training, and evaluation utilities.
*   **`he/`** – homomorphic‑encryption context management, feature encryption, and encrypted inference.

The modules are designed to be used together by the experiment scripts in the `experiments/` folder (see [Related Scripts](#related-scripts)).

## Project Structure

```
src/
├── data/
│   ├── __init__.py
│   └── loader.py              # Data loading & preprocessing
├── model/
│   ├── __init__.py
│   ├── trainer.py             # Common training routines
│   ├── labeler_a.py           # Direction A: price‑crash labels
│   ├── labeler_b.py           # Direction B: volatility labels
│   └── labeler_c.py           # Direction C: on‑chain anomaly labels
└── he/
    ├── __init__.py
    ├── encryptor.py           # HE context loading & feature encryption
    └── inference.py           # Encrypted inference for logistic regression
```

## Modules

### `data` – Data Loading & Preprocessing

**File:** `src/data/loader.py`

Handles the raw Bitcoin dataset (`BTC_DATA.csv`) and applies the same standardization that is used before homomorphic encryption.

| Function | Description |
|----------|-------------|
| `load_raw_data()` | Loads the plain‑text CSV, sorts by date, returns a `DataFrame`. |
| `load_scaler_params()` | Reads the z‑score parameters (`mean`, `std`, `fill_median`) for the 16 features from `encryption/scaler_params.csv`. |
| `preprocess_features(df, scaler)` | Applies z‑score standardization to the 16 raw features. Output shape: `(n_samples, 16)`. |
| `preprocess_chain_features(df)` | Extracts 6 on‑chain health indicators for Direction C and applies min‑max normalization. Used only for offline pseudo‑label generation. |

**Constants:**
*   `RAW_FEATURES` – list of the 16 raw feature names.
*   `CHAIN_FEATURES` – list of the 6 on‑chain features used in Direction C.

### `model` – Model Training & Label Generation

**Files:**

*   `src/model/trainer.py` – shared training utilities.
*   `src/model/labeler_a.py` – price‑crash risk labels.
*   `src/model/labeler_b.py` – volatility risk labels.
*   `src/model/labeler_c.py` – on‑chain anomaly risk labels.

#### `trainer.py`
| Function | Description |
|----------|-------------|
| `split_time_series(X, y, train_ratio=0.8)` | Splits time‑series data chronologically (no shuffling) to prevent data leakage. |
| `train_model(X_train, y_train)` | Trains a logistic‑regression classifier with `class_weight='balanced'`. Returns a fitted `LogisticRegression` object. |
| `compute_metrics(y_true, y_pred, y_prob)` | Computes accuracy, precision, recall, F1, and ROC‑AUC. |

#### Labelers
Each labeler defines a single function that returns a binary risk label (1 = high risk, 0 = low risk) for every time step.

| Direction | Function | Description |
|-----------|----------|-------------|
| **A** (price crash) | `get_crash_labels(df, horizon=14, drop_threshold=0.15)` | Labels a day as high‑risk if the minimum price in the next `horizon` days drops more than `drop_threshold`. |
| **B** (volatility) | `get_volatility_labels(df)` | Labels based on the `price30stdUSD` column: high risk if above the historical median. |
| **C** (on‑chain anomaly) | `get_anomaly_labels(X_chain, contamination=0.15, random_state=42)` | Uses Isolation Forest on the 6 on‑chain health indicators to generate pseudo‑labels. |

### `he` – Homomorphic Encryption Operations

**File:** `src/he/encryptor.py`

| Function | Description |
|----------|-------------|
| `load_public_context()` | Loads the public‑key context (no secret key) from `encryption/public_context.bin`. |
| `load_secret_context()` | Loads the secret‑key context (contains the private key) from `encryption/secret_context.bin`. |
| `encrypt_features(ctx, x_normalized)` | Encrypts a normalized 16‑dimensional feature vector into a CKKS ciphertext. |

**File:** `src/he/inference.py`

| Function | Description |
|----------|-------------|
| `he_predict_single(enc_x, weights, bias)` | Performs the linear part of logistic regression (`weights·x + bias`) on a ciphertext, decrypts the result, and applies the sigmoid. Returns `(risk_probability, inference_time_ms)`. |
| `he_predict_batch(X_normalized, weights, bias, ctx, verbose=True)` | Encrypts and infers on a batch of normalized features. Returns a dict with `probs`, `preds`, `encrypt_times_ms`, and `infer_times_ms`. |

## Usage Example

A typical workflow (as implemented in `experiments/train.py` and `experiments/infer_he.py`) looks like this:

```python
from src.data.loader import load_raw_data, load_scaler_params, preprocess_features
from src.model.labeler_a import get_crash_labels
from src.model.trainer import split_time_series, train_model
from src.he.encryptor import load_public_context, encrypt_features
from src.he.inference import he_predict_single

# 1. Load and preprocess data
df = load_raw_data()
scaler = load_scaler_params()
X = preprocess_features(df, scaler)
y = get_crash_labels(df)

# 2. Split chronologically
X_train, X_test, y_train, y_test = split_time_series(X, y)

# 3. Train a logistic‑regression model
model = train_model(X_train, y_train)
weights = model.coef_[0]
bias = model.intercept_[0]

# 4. Homomorphic encryption inference
ctx = load_public_context()
enc_x = encrypt_features(ctx, X_test[0])          # encrypt first test sample
prob, time_ms = he_predict_single(enc_x, weights, bias)
print(f"Risk probability: {prob:.4f} (took {time_ms:.2f} ms)")
```

## Dependencies

The project relies on the following Python packages (see `pyproject.toml` for exact versions):

*   `numpy` – numerical arrays.
*   `pandas` – data manipulation.
*   `scikit‑learn` – logistic regression and evaluation metrics.
*   `tenseal` – CKKS homomorphic‑encryption operations.

All experiments are run with **uv** (the recommended package manager). To install the dependencies:

```bash
uv sync
```

## Related Scripts

The `experiments/` directory contains four end‑to‑end scripts that orchestrate the modules:

1.  `train.py` – trains three logistic‑regression models (one per direction) and saves the weights.
2.  `infer_plain.py` – runs plain‑text inference on the test set (baseline).
3.  `infer_he.py` – runs homomorphic‑encryption inference on the test set.
4.  `compare.py` – compares plain‑text and HE results (accuracy, F1, timing, etc.).

Refer to the docstrings in those scripts for detailed usage instructions.

## Notes

*   The **encryption keys and scaling parameters** are stored in the `encryption/` folder at the project root (outside `src/`). They must be generated before running any HE inference (see the project’s main README).
*   All time‑series splitting is done **chronologically** to avoid data leakage; random shuffling is strictly avoided.
*   Direction C uses an Isolation Forest only for **offline pseudo‑label generation**; the final HE‑compatible model is still a logistic regression.

## License

This code is part of the CS6290 Team Project (City University of Hong Kong, Fall 2025).