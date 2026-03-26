"""
方向A：价格暴跌风险标签

定义：往后看 horizon 天，若价格最小值跌幅超过 drop_threshold 则标记为高风险(1)。
y[t] = 1  如果 min(price[t+1..t+horizon]) < price[t] * (1 - drop_threshold)
y[t] = 0  否则（含末尾无法前瞻的行）
"""
import numpy as np
import pandas as pd


def get_crash_labels(
    df: pd.DataFrame,
    horizon: int = 14,
    drop_threshold: float = 0.15,
) -> np.ndarray:
    """
    从完整数据框生成暴跌风险标签。

    参数：
        df             : 按时间排序的完整数据框
        horizon        : 前瞻天数（默认14天）
        drop_threshold : 跌幅阈值（默认15%）

    返回：shape (n_samples,) 的 int 数组，1=高风险，0=低风险
    """
    prices = df["priceUSD"].ffill().to_numpy(dtype=np.float64)
    n = len(prices)
    labels = np.zeros(n, dtype=int)
    for t in range(n - horizon):
        if prices[t] > 0:
            future_min = np.min(prices[t + 1: t + 1 + horizon])
            if future_min < prices[t] * (1 - drop_threshold):
                labels[t] = 1
    return labels
