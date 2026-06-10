# -*- coding: utf-8 -*-
"""汇总对比 4 个模型 (SVM / RF / ResNet50 / EfficientViT-M2)。

读 results/ 下 4 个结果 JSON, 生成:
  - results/all_models_comparison.csv      (各模型指标 + 相对 RF 提升)
  - results/all_models_f1_comparison.png   (F1(saline) 条形图 + 目标线)
  - results/confusion_matrix_all.png       (2x2 混淆矩阵网格)

注意 (诚实): SVM/RF 的混淆矩阵在 50 个 fold (5x10) 上累加, ResNet/EfficientViT
在 15 个 fold (5x3) 上累加 -> 绝对计数尺度不同, 各子图标题已标注累加折数;
指标 (F1 等) 已是各自折内平均, 可直接比较。

用法: conda run -n saline python src/eval/01_compare_all_models.py
"""
import sys
import json
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from config import RESULTS_DIR

TARGET_F1 = 0.65
METRICS = ["accuracy", "f1", "f1_macro", "precision", "recall"]

# (显示名, 文件, scheme 键 (None=DL 结构), CV 描述)
MODELS = [
    ("SVM", "svm_results_repeated.json", "scheme_A", "5x10 nested"),
    ("RF", "rf_results.json", "scheme_A", "5x10 fixed"),
    ("ResNet50", "resnet50_results.json", None, "5x3 repeated"),
    ("EfficientViT-M2", "efficientvit_results.json", None, "5x3 repeated"),
]


def load(file: str, scheme):
    """返回 (aggregate dict, 累加混淆矩阵 2x2, fold 数)。"""
    d = json.load(open(RESULTS_DIR / file, encoding="utf-8"))
    if scheme:  # SVM / RF
        node = d[scheme]
        agg = node["aggregate"]
        cm = np.array(node["confusion_total"], dtype=int)
        n_folds = 50  # 5x10
    else:       # ResNet / EfficientViT
        agg = d["aggregate"]
        cm = sum(np.array(f["confusion"], dtype=int) for f in d["folds"])
        n_folds = len(d["folds"])
    return agg, cm, n_folds


def main() -> None:
    data = {name: load(file, scheme) for name, file, scheme, _ in MODELS}
    cv_desc = {name: cv for name, _, _, cv in MODELS}
    rf_f1 = data["RF"][0]["f1"]["mean"]

    # ---- CSV ----
    rows = []
    for name, _, _, _ in MODELS:
        agg, _, nf = data[name]
        f1 = agg["f1"]
        rows.append({
            "model": name,
            "cv": cv_desc[name],
            "f1_mean": round(f1["mean"], 4),
            "f1_std": round(f1["std"], 4),
            "f1_ci_low": round(f1["mean"] - f1["ci95"], 4),
            "f1_ci_high": round(f1["mean"] + f1["ci95"], 4),
            "accuracy": round(agg["accuracy"]["mean"], 4),
            "f1_macro": round(agg["f1_macro"]["mean"], 4),
            "precision": round(agg["precision"]["mean"], 4),
            "recall": round(agg["recall"]["mean"], 4),
            "improvement_vs_rf_pct": round(100 * (f1["mean"] - rf_f1) / rf_f1, 1),
        })
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "all_models_comparison.csv", index=False, encoding="utf-8-sig")

    # ---- 控制台 ----
    print("=" * 78)
    print("4 模型对比 (F1 = saline 类)")
    print("=" * 78)
    print(f"{'model':<17}{'cv':<14}{'F1 [95% CI]':<24}{'Acc':<7}{'Prec':<7}{'Rec':<7}{'vs RF'}")
    for r in rows:
        ci = f"{r['f1_mean']:.3f} [{r['f1_ci_low']:.3f},{r['f1_ci_high']:.3f}]"
        print(f"{r['model']:<17}{r['cv']:<14}{ci:<24}{r['accuracy']:<7.3f}"
              f"{r['precision']:<7.3f}{r['recall']:<7.3f}{r['improvement_vs_rf_pct']:+.1f}%")
    print("-" * 78)
    print(f"目标 F1 ≥ {TARGET_F1} (RF baseline +10%)")
    for r in rows:
        ok = "✅" if r["f1_mean"] >= TARGET_F1 else "  "
        print(f"  {ok} {r['model']:<17} F1={r['f1_mean']:.3f}")

    # ---- F1 条形图 ----
    fig, ax = plt.subplots(figsize=(8, 5))
    names = [r["model"] for r in rows]
    means = [r["f1_mean"] for r in rows]
    errs = [data[n][0]["f1"]["ci95"] for n in names]
    colors = ["#9ca3af", "#9ca3af", "#4f46e5", "#0891b2"]
    bars = ax.bar(names, means, yerr=errs, capsize=6, color=colors)
    ax.axhline(TARGET_F1, ls="--", color="crimson", label=f"target F1={TARGET_F1}")
    ax.axhline(rf_f1, ls=":", color="gray", label=f"RF baseline={rf_f1:.3f}")
    for b, m in zip(bars, means):
        ax.text(b.get_x() + b.get_width() / 2, m + 0.01, f"{m:.3f}",
                ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_ylabel("F1 (saline)  mean ± 95% CI")
    ax.set_title("Model comparison: F1 (saline class)")
    ax.set_ylim(0, max(means) + max(errs) + 0.1)
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "all_models_f1_comparison.png", dpi=120)
    plt.close(fig)

    # ---- 混淆矩阵 2x2 网格 ----
    grid = [("SVM", 0, 0), ("RF", 0, 1), ("ResNet50", 1, 0), ("EfficientViT-M2", 1, 1)]
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    for name, i, j in grid:
        agg, cm, nf = data[name]
        ax = axes[i][j]
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                    xticklabels=["Pred 0", "Pred 1"], yticklabels=["True 0", "True 1"])
        ax.set_title(f"{name}  (F1={agg['f1']['mean']:.3f}, {nf}-fold accum)")
    fig.suptitle("Confusion matrices (fold-accumulated counts)", fontsize=13)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "confusion_matrix_all.png", dpi=120)
    plt.close(fig)

    print("-" * 78)
    print(f"输出: {RESULTS_DIR / 'all_models_comparison.csv'}")
    print(f"输出: {RESULTS_DIR / 'all_models_f1_comparison.png'}")
    print(f"输出: {RESULTS_DIR / 'confusion_matrix_all.png'}")


if __name__ == "__main__":
    main()
