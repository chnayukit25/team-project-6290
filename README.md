# HEBRABT: Homomorphic Encryption for Bitcoin Risk Assessment

**HEBRABT** (Homomorphic Encryption for Bitcoin Risk Assessment) is a course project for City University of Hong Kong CS6290 (Privacy‑Preserving Techniques) that implements **logistic‑regression risk prediction with full homomorphic encryption (FHE)**. The system predicts three distinct Bitcoin risk directions entirely on encrypted data using the **CKKS scheme** and the TenSEAL library, demonstrating how sensitive financial data can be analyzed without ever being decrypted.

## 📋 Overview

The project addresses three Bitcoin risk‑prediction tasks:

| Direction | Risk Type | Description |
|-----------|-----------|-------------|
| **A** | Price Crash | Predict whether the Bitcoin price will drop more than 15% within the next 14 days. |
| **B** | Volatility | Identify days with abnormally high price volatility (based on the 30‑day rolling standard deviation). |
| **C** | On‑Chain Anomaly | Detect unusual patterns in on‑chain health indicators using Isolation‑Forest pseudo‑labels. |

For each direction, a logistic‑regression model is trained on plaintext data and then applied **homomorphically** to encrypted feature vectors. The workflow compares plaintext inference (baseline) with homomorphic‑encrypted inference, measuring accuracy, F1, ROC‑AUC, prediction agreement, probability error, and inference time.

## 🗂️ Project Structure

```
team‑project‑6290/
├── encryption/                  # CKKS encryption keys, manifest, and preprocessed dataset
│   ├── BTC_DATA_he_index.csv    # Index table with placeholders for encrypted columns
│   ├── feature_manifest.json    # Encryption scheme metadata
│   ├── scaler_params.csv        # Z‑score normalization parameters (mean, std, fill‑median)
│   ├── public_context.bin       # Public encryption context (shareable)
│   ├── secret_context.bin       # Private decryption context (keep secret!)
│   ├── explore_encrypted_data.ipynb  # Jupyter notebook exploring the encrypted dataset
│   └── 说明.pdf                 # Original Chinese documentation
├── src/                         # Core Python modules
│   ├── data/                    # Data loading and preprocessing
│   ├── model/                   | Label generation, training, and evaluation
│   └── he/                      | Homomorphic‑encryption context, encryption, and inference
├── experiments/                 # End‑to‑end pipeline scripts
│   ├── train.py                 # Train three logistic‑regression models
│   ├── infer_plain.py           # Plaintext inference on test set
│   ├── infer_he.py              # Homomorphic‑encrypted inference (100 samples per direction)
│   └── compare.py               # Compare plain vs. HE results
├── models/                      # Saved model weights (`.npz` files)
├── results/                     # Inference outputs (`.csv` files)
├── outputs/                     # Aggregated metrics and comparison reports
├── BTC_DATA.csv                 # Raw Bitcoin dataset (4,389 rows × 744 columns)
├── pyproject.toml               | Project metadata and dependencies
└── README.md                    # This file
```

## 🚀 Installation

The project uses **[uv](https://github.com/astral-sh/uv)** as the package manager. Ensure you have Python ≥3.11 installed.

1. **Clone the repository** (if not already done):
   ```bash
   git clone <repository-url>
   cd team-project-6290
   ```

2. **Sync the environment** (installs all dependencies):
   ```bash
   uv sync
   ```

### Dependencies

- `numpy` – numerical arrays
- `pandas` – data manipulation
- `scikit‑learn` – logistic regression and evaluation metrics
- `tenseal` – CKKS homomorphic‑encryption operations

All dependencies are listed in `pyproject.toml` and will be installed automatically by `uv sync`.

## 📊 Usage

The pipeline is designed to be run in a fixed order. All commands should be executed from the project root.

### 1. Train the Models
Train three logistic‑regression models (one per risk direction) and save their weights to `models/`.
```bash
uv run python experiments/train.py
```
**Outputs:**
- `models/model_{a,b,c}.npz` – serialized model weights and bias
- `outputs/train_metrics.csv` – training and test performance metrics

### 2. Plaintext Inference
Run inference on the entire test set using the trained models (baseline).
```bash
uv run python experiments/infer_plain.py
```
**Outputs:**
- `results/plain_{a,b,c}.csv` – true labels, predicted probabilities, predictions, and inference times

### 3. Homomorphic‑Encrypted Inference
Perform inference on **100 encrypted samples** per direction (limited for runtime reasons).
```bash
uv run python experiments/infer_he.py
```
**Outputs:**
- `results/he_{a,b,c}.csv` – same columns as plaintext results, plus encryption times

### 4. Compare Results
Generate a comprehensive comparison of plaintext vs. HE inference.
```bash
uv run python experiments/compare.py
```
**Outputs:**
- `outputs/comparison_report.csv` – accuracy, F1, ROC‑AUC, prediction agreement, probability error, and timing breakdown

### Pipeline Summary
```
train.py → infer_plain.py → infer_he.py → compare.py
```

## 📈 Results

The comparison report (`outputs/comparison_report.csv`) shows that homomorphic‑encrypted inference achieves **identical prediction accuracy** to plaintext inference across all three directions, with negligible probability error (≤2×10⁻⁸). Inference times are orders of magnitude slower (≈4.6 ms per sample vs. ≈0.001 ms), which is expected for FHE.

| Direction | Plain Accuracy | HE Accuracy | Prediction Agreement | HE Total Time (ms) |
|-----------|----------------|-------------|----------------------|--------------------|
| A (Price Crash) | 0.57 | 0.57 | 1.0 | 4.58 |
| B (Volatility)  | 1.0 | 1.0 | 1.0 | 4.59 |
| C (On‑Chain)    | 0.86 | 0.86 | 1.0 | 4.60 |

Detailed training metrics are available in `outputs/train_metrics.csv`.

## 🔐 Encryption Details

### Dataset Encryption
The raw Bitcoin dataset (`BTC_DATA.csv`) has been pre‑encrypted using the **CKKS (Cheon‑Kim‑Kim‑Song) scheme** via TenSEAL. Sensitive columns (16 core financial metrics) are replaced with the placeholder `[HE_ENCRYPTED]` in the index table (`BTC_DATA_he_index.csv`). Each row’s encrypted features are stored as a separate ciphertext vector (not included in this repository).

### Technical Parameters (from `feature_manifest.json`)
- **Scheme**: CKKS (approximate arithmetic for real numbers)
- **Library**: TenSEAL
- **Poly modulus degree**: 8,192
- **Coefficient modulus bit sizes**: [60, 40, 40, 60]
- **Global scale**: 1,099,511,627,776 (≈2⁴⁰)
- **Encrypted features**: `priceUSD`, `transactions`, `size`, `sentbyaddress`, `difficulty`, `hashrate`, `mining_profitability`, `sentinusdUSD`, `transactionfeesUSD`, `median_transaction_feeUSD`, `confirmationtime`, `transactionvalueUSD`, `mediantransactionvalueUSD`, `activeaddresses`, `top100cap`, `fee_to_rewardUSD`

### Preprocessing
Each feature is standardized using the z‑score parameters in `encryption/scaler_params.csv`:
```
normalized_value = (raw_value - mean) / std
```
Missing values are filled with the column’s median (`fill_median`). The same transformation is applied in `src/data/loader.py` to ensure consistency between plaintext and encrypted inference.

### Security Notes
- **`public_context.bin`** can be freely shared; it allows homomorphic operations but **not decryption**.
- **`secret_context.bin`** contains the private key – **never share this file**.
- The encrypted dataset is intended for research and demonstration of privacy‑preserving technologies. Use real financial data only in compliance with applicable regulations.

## 👥 Team & Attribution

This project was developed by **Team 3** of **City University of Hong Kong CS6290 (Privacy‑Preserving Techniques), Fall 2025**.

- **Course**: CS6290 – Privacy‑Preserving Techniques
- **Institution**: City University of Hong Kong
- **Semester**: Fall 2025
- **Team**: 3

### Acknowledgements
- **CKKS Scheme**: Cheon, J. H., Kim, A., Kim, M., & Song, Y. (2017). *Homomorphic encryption for arithmetic of approximate numbers.* ASIACRYPT 2017.
- **TenSEAL**: OpenMined’s Tensorflow‑like library for homomorphic encryption.
- **Bitcoin Dataset**: Publicly available historical Bitcoin metrics.

## 📄 License

This project is intended for academic and research purposes. All code is provided as‑is under the terms of the course project. Please respect the intellectual property of the dataset and encryption keys.
