"""
共用训练器（三方向实验共用）

- split_time_series() : 按时间顺序划分训练/测试集（不打乱）
- train_model()       : 训练逻辑回归，返回模型对象
- compute_metrics()   : 计算分类评估指标
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def split_time_series(
    X: np.ndarray,
    y: np.ndarray,
    train_ratio: float = 0.8,
):
    """
    按时间顺序划分，前 train_ratio 为训练集，后面为测试集，不打乱顺序。
    时序数据严禁随机打乱，否则会产生未来数据泄露（data leakage）。
    """
    split = int(len(X) * train_ratio)
    return X[:split], X[split:], y[:split], y[split:]


def train_model(X_train: np.ndarray, y_train: np.ndarray) -> LogisticRegression:
    """
    训练逻辑回归分类器。
    class_weight='balanced' 自动处理方向A/C的标签不平衡问题。
    返回训练好的模型，coef_ 和 intercept_ 将用于 HE 推理。
    """
    model = LogisticRegression(
        C=1.0,
        max_iter=1000,
        solver="lbfgs",
        class_weight="balanced",
        random_state=42,
    )
    model.fit(X_train, y_train)
    return model


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> dict:
    """
    计算分类评估指标。
    返回包含 accuracy / precision / recall / f1 / roc_auc 的字典。
    """
    return {
        "accuracy":  round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall":    round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1":        round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "roc_auc":   round(float(roc_auc_score(y_true, y_prob)), 4),
    }
