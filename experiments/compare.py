"""
对比分析阶段：读取明文推理和HE推理的结果，计算对比指标。

对比维度：
  1. 模型性能对比：明文 vs HE 的 accuracy / f1 / roc_auc
  2. 预测一致性：两者预测标签相同的比例
  3. 概率误差：HE概率 vs 明文概率的 MAE / 最大误差
  4. 耗时对比：明文推理耗时 vs HE推理耗时

输出：
  outputs/comparison_report.csv — 完整对比报告

运行：
  uv run python experiments/compare.py
  （需先运行 train.py, infer_plain.py, infer_he.py）
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

RESULTS_DIR = Path(__file__).parent.parent / "results"
OUTPUTS_DIR = Path(__file__).parent.parent / "outputs"
DIRECTIONS = [
    ("A 暴跌风险",  "a"),
    ("B 波动率风险", "b"),
    ("C 链上异常",  "c"),
]


def compare_direction(name: str, key: str) -> dict:
    """读取一个方向的明文/HE结果，计算对比指标。"""
    plain_path = RESULTS_DIR / f"plain_{key}.csv"
    he_path    = RESULTS_DIR / f"he_{key}.csv"

    if not plain_path.exists() or not he_path.exists():
        raise FileNotFoundError(
            f"找不到结果文件，请先运行 infer_plain.py 和 infer_he.py\n"
            f"  期望: {plain_path}\n  期望: {he_path}"
        )

    plain_df = pd.read_csv(plain_path)
    he_df    = pd.read_csv(he_path)

    # 对齐到 HE 推理的样本数（HE只跑了前N条）
    n = len(he_df)
    plain_sub = plain_df.iloc[:n]

    plain_probs = plain_sub["prob"].to_numpy()
    he_probs    = he_df["prob"].to_numpy()
    plain_preds = plain_sub["pred"].to_numpy()
    he_preds    = he_df["pred"].to_numpy()
    true_labels = he_df["true_label"].to_numpy()

    # 性能指标
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    plain_acc = round(float(accuracy_score(true_labels, plain_preds)), 4)
    he_acc    = round(float(accuracy_score(true_labels, he_preds)), 4)
    plain_f1  = round(float(f1_score(true_labels, plain_preds, zero_division=0)), 4)
    he_f1     = round(float(f1_score(true_labels, he_preds,    zero_division=0)), 4)
    plain_auc = round(float(roc_auc_score(true_labels, plain_probs)), 4)
    he_auc    = round(float(roc_auc_score(true_labels, he_probs)),    4)

    # 预测一致性
    agreement = round(float((plain_preds == he_preds).mean()), 4)

    # 概率误差
    errors = np.abs(he_probs - plain_probs)
    mae       = round(float(errors.mean()), 8)
    max_error = round(float(errors.max()),  8)

    # 耗时对比
    plain_time = round(float(plain_sub["inference_time_ms"].mean()), 4)
    he_enc_time  = round(float(he_df["encrypt_time_ms"].mean()), 2)
    he_inf_time  = round(float(he_df["infer_time_ms"].mean()), 2)
    he_total_time = round(he_enc_time + he_inf_time, 2)

    print(f"\n{'='*55}")
    print(f"  方向{name} | 对比 {n} 条样本")
    print(f"{'='*55}")
    print(f"  {'':20s}  {'明文':>10s}  {'HE':>10s}")
    print(f"  {'Accuracy':20s}  {plain_acc:>10.4f}  {he_acc:>10.4f}")
    print(f"  {'F1':20s}  {plain_f1:>10.4f}  {he_f1:>10.4f}")
    print(f"  {'ROC-AUC':20s}  {plain_auc:>10.4f}  {he_auc:>10.4f}")
    print(f"  {'预测一致率':20s}  {'':>10s}  {agreement:>10.4f}")
    print(f"  {'概率MAE':20s}  {'':>10s}  {mae:>10.8f}")
    print(f"  {'概率最大误差':20s}  {'':>10s}  {max_error:>10.8f}")
    print(f"  {'平均推理耗时(ms)':20s}  {plain_time:>10.4f}  {he_total_time:>10.2f}")

    return {
        "direction":       name,
        "n_samples":       n,
        "plain_accuracy":  plain_acc,
        "he_accuracy":     he_acc,
        "plain_f1":        plain_f1,
        "he_f1":           he_f1,
        "plain_roc_auc":   plain_auc,
        "he_roc_auc":      he_auc,
        "prediction_agreement": agreement,
        "prob_mae":        mae,
        "prob_max_error":  max_error,
        "plain_time_ms":   plain_time,
        "he_encrypt_ms":   he_enc_time,
        "he_infer_ms":     he_inf_time,
        "he_total_ms":     he_total_time,
    }


def main():
    print("=" * 55)
    print("  BTC 风险评估 | 对比分析阶段")
    print("=" * 55)

    OUTPUTS_DIR.mkdir(exist_ok=True)

    rows = []
    for name, key in DIRECTIONS:
        rows.append(compare_direction(name, key))

    report_df = pd.DataFrame(rows)
    out_path = OUTPUTS_DIR / "comparison_report.csv"
    report_df.to_csv(out_path, index=False)

    print(f"\n\n对比报告已保存: {out_path}")
    print("\n核心指标摘要：")
    cols = ["direction", "plain_f1", "he_f1", "prediction_agreement",
            "prob_mae", "plain_time_ms", "he_total_ms"]
    print(report_df[cols].to_string(index=False))


if __name__ == "__main__":
    main()
