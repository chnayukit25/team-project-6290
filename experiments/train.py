"""
训练阶段：对三个方向各训练一个逻辑回归模型，保存权重与训练指标。

输出：
  models/model_a.npz  — 方向A权重 + 元数据
  models/model_b.npz  — 方向B权重 + 元数据
  models/model_c.npz  — 方向C权重 + 元数据
  outputs/train_metrics.csv — 三方向训练/测试集指标汇总

运行：
  uv run python experiments/train.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# 将项目根目录加入路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import (
    load_raw_data,
    load_scaler_params,
    preprocess_chain_features,
    preprocess_features,
)
from src.model.labeler_a import get_crash_labels
from src.model.labeler_b import get_volatility_labels
from src.model.labeler_c import get_anomaly_labels
from src.model.trainer import compute_metrics, split_time_series, train_model

MODELS_DIR = Path(__file__).parent.parent / "models"
OUTPUTS_DIR = Path(__file__).parent.parent / "outputs"


def train_direction(name: str, X: np.ndarray, y: np.ndarray) -> dict:
    """训练单个方向，返回指标字典，保存模型权重。"""
    print(f"\n{'='*50}")
    print(f"  方向{name} | 高风险={y.sum()} ({100*y.mean():.1f}%) 低风险={(1-y).sum()}")
    print(f"{'='*50}")

    X_train, X_test, y_train, y_test = split_time_series(X, y)
    print(f"  训练集: {len(X_train)} 条  测试集: {len(X_test)} 条")

    model = train_model(X_train, y_train)
    weights = model.coef_[0]        # shape (16,)
    bias = float(model.intercept_[0])

    # 训练集指标
    y_train_pred = model.predict(X_train)
    y_train_prob = model.predict_proba(X_train)[:, 1]
    train_metrics = compute_metrics(y_train, y_train_pred, y_train_prob)

    # 测试集指标
    y_test_pred = model.predict(X_test)
    y_test_prob = model.predict_proba(X_test)[:, 1]
    test_metrics = compute_metrics(y_test, y_test_pred, y_test_prob)

    print(f"  [训练集] acc={train_metrics['accuracy']}  f1={train_metrics['f1']}  auc={train_metrics['roc_auc']}")
    print(f"  [测试集] acc={test_metrics['accuracy']}  f1={test_metrics['f1']}  auc={test_metrics['roc_auc']}")

    # 保存模型权重（npz格式：weights + bias + test_indices + label_counts）
    direction_key = name.split()[0].lower()
    model_path = MODELS_DIR / f"model_{direction_key}.npz"
    train_size = len(X_train)
    np.savez(
        model_path,
        weights=weights,
        bias=np.array([bias]),
        train_size=np.array([train_size]),
        label_counts=np.array([int((1 - y).sum()), int(y.sum())]),  # [低风险, 高风险]
    )
    print(f"  模型已保存: {model_path}")

    return [
        {"direction": name, "split": "train", **train_metrics},
        {"direction": name, "split": "test",  **test_metrics},
    ]


def main():
    print("=" * 55)
    print("  BTC 风险评估 | 训练阶段")
    print("=" * 55)

    MODELS_DIR.mkdir(exist_ok=True)
    OUTPUTS_DIR.mkdir(exist_ok=True)

    df = load_raw_data()
    scaler = load_scaler_params()
    X = preprocess_features(df, scaler)
    X_chain = preprocess_chain_features(df)

    all_metrics = []

    # 方向A：价格暴跌风险
    y_a = get_crash_labels(df)
    all_metrics.extend(train_direction("A 暴跌风险", X, y_a))

    # 方向B：价格波动率风险
    y_b = get_volatility_labels(df)
    all_metrics.extend(train_direction("B 波动率风险", X, y_b))

    # 方向C：链上异常风险
    y_c = get_anomaly_labels(X_chain)
    all_metrics.extend(train_direction("C 链上异常", X, y_c))

    # 保存汇总指标
    metrics_df = pd.DataFrame(all_metrics)
    metrics_path = OUTPUTS_DIR / "train_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"\n训练指标已保存: {metrics_path}")

    print("\n\n训练指标汇总：")
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
