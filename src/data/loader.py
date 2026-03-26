"""
数据加载与预处理模块（三方向实验共用）

- load_raw_data()         : 加载明文 BTC_DATA.csv
- load_scaler_params()    : 加载16个特征的 z-score 参数
- preprocess_features()   : 对16个原始特征做标准化（与HE加密前一致）
- preprocess_chain_features(): 提取链上健康指标（方向C用）
"""
from pathlib import Path

import numpy as np
import pandas as pd

DATA_ROOT = Path(__file__).parent.parent.parent

# 16个原始特征（对应 HE 加密字段，顺序与 scaler_params.csv 一致）
RAW_FEATURES = [
    "priceUSD",
    "transactions",
    "size",
    "sentbyaddress",
    "difficulty",
    "hashrate",
    "mining_profitability",
    "sentinusdUSD",
    "transactionfeesUSD",
    "median_transaction_feeUSD",
    "confirmationtime",
    "transactionvalueUSD",
    "mediantransactionvalueUSD",
    "activeaddresses",
    "top100cap",
    "fee_to_rewardUSD",
]

# 方向C使用的链上健康指标（明文列，仅用于生成异常伪标签）
CHAIN_FEATURES = [
    "confirmationtime",
    "transactionfeesUSD",
    "activeaddresses",
    "hashrate",
    "transactions",
    "fee_to_rewardUSD",
]


def load_raw_data() -> pd.DataFrame:
    """加载原始明文数据，按日期升序排列。"""
    df = pd.read_csv(DATA_ROOT / "BTC_DATA.csv", index_col=0)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    return df


def load_scaler_params() -> pd.DataFrame:
    """加载16个特征的标准化参数（fill_median / mean / std）。"""
    return pd.read_csv(
        DATA_ROOT / "encryption" / "scaler_params.csv",
        index_col="feature",
    )


def preprocess_features(df: pd.DataFrame, scaler: pd.DataFrame) -> np.ndarray:
    """
    对16个原始特征做 z-score 标准化，与 HE 加密前的处理保持一致。
    NaN 先用 fill_median 填充，再做 (x - mean) / std。
    返回 shape (n_samples, 16) 的 float64 数组。
    """
    X = df[RAW_FEATURES].copy()
    for feat in RAW_FEATURES:
        p = scaler.loc[feat]
        X[feat] = X[feat].fillna(p["fill_median"])
        X[feat] = (X[feat] - p["mean"]) / p["std"]
    return X.to_numpy(dtype=np.float64)


def preprocess_chain_features(df: pd.DataFrame) -> np.ndarray:
    """
    方向C专用：提取链上健康指标并做 min-max 标准化。
    仅用于离线生成异常伪标签，不参与 HE 推理。
    返回 shape (n_samples, 6) 的 float64 数组。
    """
    X = df[CHAIN_FEATURES].copy()
    X = X.fillna(X.median())
    X = (X - X.min()) / (X.max() - X.min() + 1e-9)
    return X.to_numpy(dtype=np.float64)
