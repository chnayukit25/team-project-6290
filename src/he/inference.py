"""
HE 密文推理模块

核心逻辑：
  逻辑回归推理 = sigmoid( w·x + b )
  其中 w·x + b 全程在密文上完成（CKKS 加法+乘法），sigmoid 在解密后做。

- he_predict_single() : 对单条密文做推理，返回风险概率
- he_predict_batch()  : 对多行明文数据逐行加密推理，返回概率数组
"""
import time

import numpy as np
import tenseal as ts

from src.he.encryptor import encrypt_features


def he_predict_single(
    enc_x: ts.CKKSVector,
    weights: np.ndarray,
    bias: float,
) -> tuple[float, float]:
    """
    在密文上运行逻辑回归线性层，解密后应用 sigmoid。

    参数：
        enc_x   : 加密的16维特征向量
        weights : 模型权重 (16,)，明文
        bias    : 模型偏置，明文

    返回：(风险概率, 推理耗时ms) 的元组
    """
    t0 = time.perf_counter()
    enc_logit = enc_x.dot(weights.tolist())
    enc_logit += bias
    logit = enc_logit.decrypt()[0]
    prob = float(1.0 / (1.0 + np.exp(-logit)))
    elapsed_ms = (time.perf_counter() - t0) * 1000
    return prob, elapsed_ms


def he_predict_batch(
    X_normalized: np.ndarray,
    weights: np.ndarray,
    bias: float,
    ctx: ts.Context,
    verbose: bool = True,
) -> dict:
    """
    对多行数据逐行加密并推理。

    参数：
        X_normalized : shape (n_samples, 16)，已标准化的明文特征
        weights      : 模型权重 (16,)
        bias         : 模型偏置
        ctx          : TenSEAL 上下文（含公钥）
        verbose      : 是否打印进度

    返回：字典，包含：
        probs            : shape (n_samples,) 风险概率
        preds            : shape (n_samples,) 预测标签（>=0.5 为1）
        encrypt_times_ms : 每条加密耗时
        infer_times_ms   : 每条推理耗时
    """
    n = len(X_normalized)
    probs = np.zeros(n)
    encrypt_times = np.zeros(n)
    infer_times = np.zeros(n)

    for i, x in enumerate(X_normalized):
        t_enc = time.perf_counter()
        enc_x = encrypt_features(ctx, x)
        encrypt_times[i] = (time.perf_counter() - t_enc) * 1000

        prob, infer_ms = he_predict_single(enc_x, weights, bias)
        probs[i] = prob
        infer_times[i] = infer_ms

        if verbose and (i + 1) % 20 == 0:
            print(f"    进度: {i+1}/{n}")

    return {
        "probs": probs,
        "preds": (probs >= 0.5).astype(int),
        "encrypt_times_ms": encrypt_times,
        "infer_times_ms": infer_times,
    }
