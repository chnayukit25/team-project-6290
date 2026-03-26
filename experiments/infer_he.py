"""
HE 密文推理阶段：加载模型权重，对测试集特征加密后在密文上推理。

流程：明文特征 → 加密(公钥) → 密文上线性运算 → 解密(私钥) → sigmoid → 风险概率

输出：
  results/he_a.csv — 方向A的HE推理结果
  results/he_b.csv — 方向B的HE推理结果
  results/he_c.csv — 方向C的HE推理结果
  每个CSV包含：true_label, prob, pred, encrypt_time_ms, infer_time_ms

注意：为控制运行时间，默认对测试集前 N_SAMPLES 条做 HE 推理。

运行：
  uv run python experiments/infer_he.py
  （需先运行 experiments/train.py）
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import (
    load_raw_data,
    load_scaler_params,
    preprocess_chain_features,
    preprocess_features,
)
from src.he.encryptor import encrypt_features, load_secret_context
from src.he.inference import he_predict_single
from src.model.labeler_a import get_crash_labels
from src.model.labeler_b import get_volatility_labels
from src.model.labeler_c import get_anomaly_labels
from src.model.trainer import compute_metrics

MODELS_DIR = Path(__file__).parent.parent / "models"
RESULTS_DIR = Path(__file__).parent.parent / "results"
N_SAMPLES = 100   # HE推理样本数（每条约5ms，100条约8秒）


def infer_he(name: str, X: np.ndarray, y: np.ndarray, ctx) -> dict:
    """加载权重，对测试集前 N_SAMPLES 条做 HE 推理，保存结果。"""
    direction_key = name.split()[0].lower()
    model_path = MODELS_DIR / f"model_{direction_key}.npz"

    if not model_path.exists():
        raise FileNotFoundError(f"找不到模型文件 {model_path}，请先运行 train.py")

    data = np.load(model_path)
    weights = data["weights"]
    bias = float(data["bias"][0])
    train_size = int(data["train_size"][0])

    X_test = X[train_size: train_size + N_SAMPLES]
    y_test = y[train_size: train_size + N_SAMPLES]

    print(f"\n{'='*50}")
    print(f"  HE推理 | 方向{name} | {len(X_test)} 条")
    print(f"{'='*50}")

    probs = np.zeros(len(X_test))
    encrypt_times = np.zeros(len(X_test))
    infer_times = np.zeros(len(X_test))

    for i, x in enumerate(X_test):
        # 加密
        t_enc = time.perf_counter()
        enc_x = encrypt_features(ctx, x)
        encrypt_times[i] = (time.perf_counter() - t_enc) * 1000

        # 密文推理
        prob, infer_ms = he_predict_single(enc_x, weights, bias)
        probs[i] = prob
        infer_times[i] = infer_ms

        if (i + 1) % 20 == 0:
            print(f"    进度: {i+1}/{len(X_test)}")

    preds = (probs >= 0.5).astype(int)
    metrics = compute_metrics(y_test, preds, probs)

    print(f"  acc={metrics['accuracy']}  precision={metrics['precision']}  "
          f"recall={metrics['recall']}  f1={metrics['f1']}  auc={metrics['roc_auc']}")
    print(f"  平均加密耗时: {encrypt_times.mean():.2f}ms  平均推理耗时: {infer_times.mean():.2f}ms")

    result_df = pd.DataFrame({
        "true_label":       y_test,
        "prob":             probs.round(6),
        "pred":             preds,
        "encrypt_time_ms":  encrypt_times.round(4),
        "infer_time_ms":    infer_times.round(4),
    })
    out_path = RESULTS_DIR / f"he_{direction_key}.csv"
    result_df.to_csv(out_path, index=False)
    print(f"  结果已保存: {out_path}")

    return {
        "direction": name,
        "method": "HE",
        **metrics,
        "avg_encrypt_ms": round(float(encrypt_times.mean()), 2),
        "avg_infer_ms":   round(float(infer_times.mean()), 2),
    }


def main():
    print("=" * 55)
    print("  BTC 风险评估 | HE 密文推理阶段")
    print(f"  每方向推理 {N_SAMPLES} 条样本")
    print("=" * 55)

    RESULTS_DIR.mkdir(exist_ok=True)

    print("\n加载 HE 上下文（含私钥）...")
    ctx = load_secret_context()

    df = load_raw_data()
    scaler = load_scaler_params()
    X = preprocess_features(df, scaler)
    X_chain = preprocess_chain_features(df)

    results = []
    results.append(infer_he("A 暴跌风险",  X, get_crash_labels(df),        ctx))
    results.append(infer_he("B 波动率风险", X, get_volatility_labels(df),   ctx))
    results.append(infer_he("C 链上异常",   X, get_anomaly_labels(X_chain), ctx))

    print("\n\nHE推理指标汇总：")
    print(pd.DataFrame(results).to_string(index=False))


if __name__ == "__main__":
    main()
