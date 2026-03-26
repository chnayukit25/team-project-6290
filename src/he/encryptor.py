"""
HE 上下文加载与特征加密

- load_public_context()  : 加载公钥上下文（加密用，服务端持有）
- load_secret_context()  : 加载私钥上下文（解密用，用户持有）
- encrypt_features()     : 将标准化后的16维特征向量加密为 CKKS 密文
"""
from pathlib import Path

import numpy as np
import tenseal as ts

ENCRYPTION_DIR = Path(__file__).parent.parent.parent / "encryption"


def load_public_context() -> ts.Context:
    """加载公钥上下文（不含私钥，用于加密）。"""
    with open(ENCRYPTION_DIR / "public_context.bin", "rb") as f:
        return ts.context_from(f.read())


def load_secret_context() -> ts.Context:
    """加载含私钥的上下文（用于解密结果）。"""
    with open(ENCRYPTION_DIR / "secret_context.bin", "rb") as f:
        return ts.context_from(f.read())


def encrypt_features(ctx: ts.Context, x_normalized: np.ndarray) -> ts.CKKSVector:
    """
    将归一化后的16维特征向量加密为 CKKS 密文。

    参数：
        ctx          : TenSEAL 上下文（公钥或私钥均可用于加密）
        x_normalized : shape (16,) 的 float64 数组，已经过 z-score 标准化

    返回：ts.CKKSVector 密文对象
    """
    return ts.ckks_vector(ctx, x_normalized.tolist())
