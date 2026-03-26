"""
方向C：链上异常风险标签（两阶段）

阶段1（离线）：用 Isolation Forest 在链上健康指标上检测历史异常期 → 生成伪标签
阶段2（推理）：用伪标签训练逻辑回归，权重用于 HE 推理

Isolation Forest 本身无法在密文上运行，仅在离线标签生成阶段使用。
最终推理仍由逻辑回归（HE兼容）完成。
"""
import numpy as np
from sklearn.ensemble import IsolationForest


def get_anomaly_labels(
    X_chain: np.ndarray,
    contamination: float = 0.15,
    random_state: int = 42,
) -> np.ndarray:
    """
    用 Isolation Forest 生成链上异常风险伪标签。

    参数：
        X_chain       : 链上健康指标矩阵，shape (n_samples, n_features)
                        由 loader.preprocess_chain_features() 生成
        contamination : 预期异常比例（默认15%）
        random_state  : 随机种子

    返回：shape (n_samples,) 的 int 数组，1=异常(高风险)，0=正常(低风险)
    """
    iso = IsolationForest(
        contamination=contamination,
        n_estimators=100,
        random_state=random_state,
    )
    # IsolationForest 输出：-1=异常，1=正常 → 转换为 1=高风险，0=低风险
    raw = iso.fit_predict(X_chain)
    return (raw == -1).astype(int)
