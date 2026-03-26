"""
明文推理阶段：加载训练好的模型权重，在测试集明文特征上做推理。

输出：
  results/plain_a.csv — 方向A明文推理结果
  results/plain_b.csv — 方向B明文推理结果
  results/plain_c.csv — 方向C明文推理结果
  每个CSV包含：true_label, prob, pred, inference_time_ms

运行：
  uv run python experiments/infer_plain.py
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
from src.model.labeler_a import get_crash_labels
from src.model.labeler_b import get_volatility_labels
from src.model.labeler_c import get_anomaly_labels
from src.model.trainer import compute_metrics, split_time_series

MODELS_DIR = Path(__file__).parent.parent / "models"
RESULTS_DIR = Path(__file__).parent.parent / "results"


def infer_plain(name: str, X: np.ndarray, y: np.ndarray) -> dict:
    """加载权重，对测试集做明文推理，保存结果。"""
    direction_key = name.split()[0].lower()
    model_path = MODELS_DIR / f"model_{direction_key}.npz"

    if not model_path.exists():
        raise FileNotFoundError(f"找不到模型文件 {model_path}，请先运行 train.py")

    data = np.load(model_path)
    weights = data["weights"]           # shape (16,)
    bias = float(data["bias"][0])
    train_size = int(data["train_size"][0])

    # 还原测试集（与训练时相同的划分）
    X_test = X[train_size:]
    y_test = y[train_size:]

    print(f"\n{'='*50}")
    print(f"  明文推理 | 方向{name} | 测试集 {len(X_test)} 条")
    print(f"{'='*50}")

    # 逐条计时推理（sigmoid(w·x + b)）
    probs = np.zeros(len(X_test))
    times_ms = np.zeros(len(X_test))
    for i, x in enumerate(X_test):
        t0 = time.perf_counter()
        logit = float(np.dot(weights, x) + bias)
        probs[i] = 1.0 / (1.0 + np.exp(-logit))
        times_ms[i] = (time.perf_counter() - t0) * 1000

    preds = (probs >= 0.5).astype(int)
    metrics = compute_metrics(y_test, preds, probs)

    print(f"  acc={metrics['accuracy']}  precision={metrics['precision']}  "
          f"recall={metrics['recall']}  f1={metrics['f1']}  auc={metrics['roc_auc']}")
    print(f"  平均推理耗时: {times_ms.mean():.4f}ms/条")

    # 保存结果
    result_df = pd.DataFrame({
        "true_label":        y_test,
        "prob":              probs.round(6),
        "pred":              preds,
        "inference_time_ms": times_ms.round(4),
    })
    out_path = RESULTS_DIR / f"plain_{direction_key}.csv"
    result_df.to_csv(out_path, index=False)
    print(f"  结果已保存: {out_path}")

    return {"direction": name, "method": "plaintext", **metrics,
            "avg_time_ms": round(float(times_ms.mean()), 4)}


def main():
    print("=" * 55)
    print("  BTC 风险评估 | 明文推理阶段")
    print("=" * 55)

    RESULTS_DIR.mkdir(exist_ok=True)

    df = load_raw_data()
    scaler = load_scaler_params()
    X = preprocess_features(df, scaler)
    X_chain = preprocess_chain_features(df)

    results = []
    results.append(infer_plain("A 暴跌风险",  X, get_crash_labels(df)))
    results.append(infer_plain("B 波动率风险", X, get_volatility_labels(df)))
    results.append(infer_plain("C 链上异常",   X, get_anomaly_labels(X_chain)))

    print("\n\n明文推理指标汇总：")
    print(pd.DataFrame(results).to_string(index=False))


if __name__ == "__main__":
    main()
