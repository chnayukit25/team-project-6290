# Encrypted Bitcoin Dataset – CKKS Homomorphic Encryption

This directory contains a Bitcoin dataset encrypted using the **CKKS (Cheon-Kim-Kim-Song) homomorphic encryption scheme** implemented with the TenSEAL library. The dataset enables privacy-preserving computations on sensitive financial data while maintaining the ability to perform statistical analysis and machine learning inference without decrypting the data.

## 📁 Files

| File | Description |
|------|-------------|
| `BTC_DATA_he_index.csv` | **Index table** – human-readable structure with plaintext non-sensitive columns and placeholders (`[HE_ENCRYPTED]`) for encrypted sensitive columns. Includes a `he_row_id` column that links each row to its corresponding ciphertext. |
| `explore_encrypted_data.ipynb` | Jupyter notebook demonstrating how to load, explore, and perform basic operations on the encrypted dataset. |
| `feature_manifest.json` | **Metadata manifest** – describes the encryption scheme, library, column names, encrypted features, and CKKS parameters. |
| `scaler_params.csv` | **Preprocessing parameters** – median, mean, and standard deviation for each encrypted feature, used for normalization before encryption. |
| `public_context.bin` | **Public encryption context** – can be shared with third parties to perform homomorphic operations (e.g., inference) without exposing the private key. |
| `secret_context.bin` | **Private decryption context** – contains the secret key; **keep this file secure** and never share it. |
| `说明.pdf` | Original documentation in Chinese (refer to this for detailed technical background). |

## 🔐 Encryption Scheme

### High‑Level Design
- **Row‑level indexed encryption**: Each row of sensitive data is encrypted as a single ciphertext vector. Non‑sensitive columns remain in plaintext in the index table for reference.
- **Placeholder replacement**: Sensitive columns are replaced with `[HE_ENCRYPTED]` in the CSV index table. The actual encrypted values are stored separately (not included in this release).
- **Linkage**: The `he_row_id` column uniquely identifies each row and can be used to retrieve the corresponding ciphertext from a cipher store (e.g., `cipher_store.jsonl` – not included here).

### Technical Parameters (from `feature_manifest.json`)
- **Scheme**: CKKS (approximate arithmetic for real numbers)
- **Library**: TenSEAL (Tensorflow‑like API for homomorphic encryption)
- **Index mode**: `row_level_reference`
- **Rows**: 4,389
- **Columns**: 744 total columns (see manifest for full list)
- **Encrypted features**: 16 core Bitcoin metrics (priceUSD, transactions, size, sentbyaddress, difficulty, hashrate, mining_profitability, sentinusdUSD, transactionfeesUSD, median_transaction_feeUSD, confirmationtime, transactionvalueUSD, mediantransactionvalueUSD, activeaddresses, top100cap, fee_to_rewardUSD)
- **Poly modulus degree**: 8,192
- **Coefficient modulus bit sizes**: [60, 40, 40, 60]
- **Global scale**: 1,099,511,627,776 (≈2⁴⁰)

### Preprocessing
Each sensitive feature was normalized using the parameters in `scaler_params.csv`:
```
feature, fill_median, mean, std
```
The `fill_median` value was used to impute missing entries before applying z‑score normalization:  
`normalized_value = (raw_value - mean) / std`

## 🚀 Usage

### 1. Explore the Dataset
Open `explore_encrypted_data.ipynb` in Jupyter to:
- Load the manifest and understand the encryption parameters
- Inspect the index table (`BTC_DATA_he_index.csv`)
- View the preprocessing statistics (`scaler_params.csv`)
- Load the public/secret contexts (requires TenSEAL)

### 2. Perform Homomorphic Operations
With the public context, you can:
- Add, multiply, or rotate ciphertexts (TenSEAL operations)
- Compute privacy‑preserving statistics (e.g., mean, variance)
- Run encrypted machine‑learning models (if models are adapted for CKKS)

### 3. Decrypt (If You Have the Secret Context)
**Warning**: Only decrypt if you are the data owner or have explicit permission. Use the secret context to recover the original normalized values, then reverse the scaling using `mean` and `std` from `scaler_params.csv`.

## 🧩 Example Workflow

```python
import tenseal as ts
import pandas as pd
import json

# Load public context
with open("public_context.bin", "rb") as f:
    public_context = ts.context_from(f.read())

# Load index table
df = pd.read_csv("BTC_DATA_he_index.csv")
print(df.head())

# Load manifest
with open("feature_manifest.json", "r") as f:
    manifest = json.load(f)

print(f"Encrypted features: {manifest['encrypted_features']}")
```

## 📚 References

- **CKKS Scheme**: Cheon, J. H., Kim, A., Kim, M., & Song, Y. (2017). *Homomorphic encryption for arithmetic of approximate numbers.* ASIACRYPT 2017.
- **TenSEAL**: *A library for doing homomorphic encryption operations on tensors.* [GitHub](https://github.com/OpenMined/TenSEAL)
- **Original Chinese Documentation**: See `说明.pdf` for detailed technical background and project context.

## ⚠️ Security Notes

- **Never share `secret_context.bin`** – it contains the private decryption key.
- The public context (`public_context.bin`) can be freely distributed; it allows homomorphic operations but not decryption.
- This dataset is intended for research and demonstration of privacy‑preserving technologies. Ensure compliance with data‑protection regulations when using real financial data.

---

*This dataset is part of a course project for City University of Hong Kong CS6290 (Privacy‑Preserving Techniques), Spring 2025.*