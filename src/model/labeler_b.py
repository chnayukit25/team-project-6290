"""
方向B：价格波动率风险标签

定义：使用 price30stdUSD（30日价格标准差，明文列）作为波动率代理指标。
大于历史中位数 → 高波动风险(1)，否则低风险(0)。
标签天然约50/50平衡，是三个方向中最稳定的 baseline。
"""
import numpy as np
import pandas as pd


def get_volatility_labels(df: pd.DataFrame) -> np.ndarray:
    """
    从完整数据框生成波动率风险标签。

    参数：
        df : 按时间排序的完整数据框（需包含 price30stdUSD 列）

    返回：shape (n_samples,) 的 int 数组，1=高风险，0=低风险
    """
    col = df["price30stdUSD"].fillna(0.0)
    threshold = col.median()
    return (col > threshold).astype(int).to_numpy()
