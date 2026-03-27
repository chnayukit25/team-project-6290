# Experiments

This folder contains the main pipeline scripts for BTC risk assessment using logistic regression with homomorphic encryption (HE). The experiments compare plaintext inference with HE-encrypted inference across three risk directions.

## Scripts

The pipeline must be run in the following order:

### 1. `train.py`
- **Purpose**: Train logistic regression models for three risk directions (A: price crash, B: volatility, C: on‑chain anomaly).
- **Input**: Raw BTC data (`BTC_DATA.csv`).
- **Output**: Model weights saved as `models/model_{a,b,c}.npz` and training metrics in `outputs/train_metrics.csv`.
- **Run**: `uv run python experiments/train.py`

### 2. `infer_plain.py`
- **Purpose**: Perform plaintext inference on the test set using the trained weights.
- **Prerequisite**: `train.py` must have run successfully.
- **Output**: Plain inference results saved as `results/plain_{a,b,c}.csv` (columns: `true_label`, `prob`, `pred`, `inference_time_ms`).
- **Run**: `uv run python experiments/infer_plain.py`

### 3. `infer_he.py`
- **Purpose**: Perform homomorphic‑encrypted inference on a subset of the test set (default 100 samples).
- **Prerequisite**: `train.py` must have run successfully.
- **Output**: HE inference results saved as `results/he_{a,b,c}.csv` (columns: `true_label`, `prob`, `pred`, `encrypt_time_ms`, `infer_time_ms`).
- **Run**: `uv run python experiments/infer_he.py`

### 4. `compare.py`
- **Purpose**: Compare plaintext and HE inference results across multiple dimensions (accuracy, F1, ROC‑AUC, prediction agreement, probability error, inference time).
- **Prerequisites**: Both `infer_plain.py` and `infer_he.py` must have run.
- **Output**: Comprehensive comparison report saved as `outputs/comparison_report.csv`.
- **Run**: `uv run python experiments/compare.py`

## Pipeline Order

```
train.py → infer_plain.py → infer_he.py → compare.py
```

## Output Directories

- `models/` – trained model weights (`.npz` files).
- `results/` – inference results (`.csv` files) from plain and HE runs.
- `outputs/` – aggregated metrics and comparison reports.

## Dependencies

All scripts are run with [uv](https://github.com/astral-sh/uv). Ensure the project environment is activated (`uv sync`) before executing any script.

## Notes

- HE inference is limited to 100 samples per direction to keep runtime manageable (approx. 8 seconds per direction).
- The comparison report includes performance metrics, prediction agreement, probability errors, and timing breakdowns.