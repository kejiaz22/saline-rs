# -*- coding: utf-8 -*-
"""SVM (RBF) 训练: 小样本最佳实践 (nested CV + 置信区间 + 低置信消融)。

输入: data/processed/features.csv (60 行, 34 特征)

实验:
  方案 A: 全样本 (n=60)
  方案 B: 剔除 low-confidence (n=44)

# Initial 5-fold gave wide CI [0.36, 0.83]. Upgraded to 5x10 repeated CV
# for stable estimation.
CV: RepeatedStratifiedKFold(5 splits x 10 repeats = 50 估计)。
每个 outer fold 内用 GridSearchCV(3-fold) 选超参 (nested CV, 无偏估计)。
Pipeline(StandardScaler + SVC) 保证 scaler 只在 train fold 上 fit (无泄漏)。

输出:
  - 控制台: 各指标 mean ± std + 95% CI, A vs B 对比
  - results/svm_results.json
  - results/svm_confusion_matrix.png (A/B 并排)

用法: conda run -n saline python src/classical/02_train_svm.py
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

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import RepeatedStratifiedKFold, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
)

from config import DATA_PROCESSED, RESULTS_DIR, RANDOM_SEED

FEATURES_CSV = DATA_PROCESSED / "features.csv"
META_COLS = ["patch_id", "region", "label", "confidence"]

PARAM_GRID = {
    "svc__C": [0.1, 1, 10, 100],
    "svc__gamma": ["scale", "auto", 0.01, 0.1, 1],
}
N_SPLITS = 5
N_REPEATS = 10
INNER_CV = 3


def build_pipeline() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("svc", SVC(kernel="rbf", random_state=RANDOM_SEED)),
    ])


def ci95(std: float, n: int) -> float:
    return 1.96 * std / np.sqrt(n)


def run_nested_cv(X: np.ndarray, y: np.ndarray) -> dict:
    """nested 5-fold CV, 返回逐 fold 指标 / 超参 / 累加混淆矩阵。"""
    skf = RepeatedStratifiedKFold(
        n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=RANDOM_SEED
    )
    fold_metrics = []
    fold_params = []
    cm_total = np.zeros((2, 2), dtype=int)

    for tr, va in skf.split(X, y):
        grid = GridSearchCV(
            build_pipeline(), PARAM_GRID, scoring="f1", cv=INNER_CV, n_jobs=-1,
        )
        grid.fit(X[tr], y[tr])
        best = grid.best_estimator_
        pred = best.predict(X[va])
        yt = y[va]

        fold_metrics.append({
            "accuracy": accuracy_score(yt, pred),
            "precision": precision_score(yt, pred, pos_label=1, zero_division=0),
            "recall": recall_score(yt, pred, pos_label=1, zero_division=0),
            "f1": f1_score(yt, pred, pos_label=1, zero_division=0),
            "f1_macro": f1_score(yt, pred, average="macro", zero_division=0),
        })
        fold_params.append({
            "C": grid.best_params_["svc__C"],
            "gamma": grid.best_params_["svc__gamma"],
        })
        cm_total += confusion_matrix(yt, pred, labels=[0, 1])

    # 聚合
    keys = fold_metrics[0].keys()
    agg = {}
    n_est = len(fold_metrics)  # 50
    for k in keys:
        vals = np.array([m[k] for m in fold_metrics])
        std = float(np.std(vals, ddof=1))
        agg[k] = {
            "mean": float(np.mean(vals)),
            "std": std,
            "ci95": ci95(std, n_est),
        }

    # 超参众数
    param_strs = [f"C={p['C']}, gamma={p['gamma']}" for p in fold_params]
    mode_param = max(set(param_strs), key=param_strs.count)

    return {
        "n": int(len(y)),
        "fold_metrics": fold_metrics,
        "fold_params": fold_params,
        "mode_param": mode_param,
        "aggregate": agg,
        "confusion_total": cm_total.tolist(),
    }


def print_scheme(title: str, res: dict) -> None:
    a = res["aggregate"]
    print(f"【{title} (n={res['n']})】")
    # 50 个 fold 的超参分布 (而非逐个列出)
    from collections import Counter
    pstrs = [f"C={p['C']}, gamma={p['gamma']}" for p in res["fold_params"]]
    dist = Counter(pstrs).most_common()
    print(f"  超参选择分布 ({len(pstrs)} folds): "
          + "; ".join(f"[{s}]x{c}" for s, c in dist[:4]))
    print(f"  众数超参    : {res['mode_param']}")
    for key, label in [("accuracy", "Accuracy"), ("f1", "F1 (saline)"),
                       ("f1_macro", "F1 macro"), ("precision", "Precision(1)"),
                       ("recall", "Recall(1)")]:
        m = a[key]
        print(f"  {label:13s}: {m['mean']:.3f} ± {m['std']:.3f}  "
              f"(95% CI: [{m['mean']-m['ci95']:.3f}, {m['mean']+m['ci95']:.3f}])")
    cm = res["confusion_total"]
    print("  Confusion matrix (5-fold 累加):")
    print("                 Pred 0  Pred 1")
    print(f"        True 0:   {cm[0][0]:4d}    {cm[0][1]:4d}")
    print(f"        True 1:   {cm[1][0]:4d}    {cm[1][1]:4d}")
    print()


def plot_confusions(resA: dict, resB: dict, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, res, title in [
        (axes[0], resA, f"Scheme A: all (n={resA['n']})"),
        (axes[1], resB, f"Scheme B: no low-conf (n={resB['n']})"),
    ]:
        cm = np.array(res["confusion_total"])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                    xticklabels=["Pred 0", "Pred 1"],
                    yticklabels=["True 0", "True 1"])
        ax.set_title(f"{title}\nF1(saline)={res['aggregate']['f1']['mean']:.3f}")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main() -> None:
    df = pd.read_csv(FEATURES_CSV)
    feat_cols = [c for c in df.columns if c not in META_COLS]

    # 方案 A: 全样本
    Xa = df[feat_cols].values
    ya = df["label"].values.astype(int)
    print("=" * 70)
    print("SVM 训练结果")
    print("=" * 70 + "\n")
    resA = run_nested_cv(Xa, ya)
    print_scheme("方案 A: 全样本", resA)

    # 方案 B: 剔除 low-confidence
    dfB = df[df["confidence"].isin(["high", "medium"])].reset_index(drop=True)
    Xb = dfB[feat_cols].values
    yb = dfB["label"].values.astype(int)
    resB = run_nested_cv(Xb, yb)
    print_scheme("方案 B: 剔除 low-confidence", resB)

    # 对比
    fa = resA["aggregate"]["f1"]["mean"]
    fb = resB["aggregate"]["f1"]["mean"]
    print("【对比】")
    print(f"  方案 A F1(saline): {fa:.3f}")
    print(f"  方案 B F1(saline): {fb:.3f}")
    print(f"  差异 (B - A)     : {fb - fa:+.3f}")
    overlap = (abs(fb - fa) < resA["aggregate"]["f1"]["ci95"] + resB["aggregate"]["f1"]["ci95"])
    print(f"  解读: 剔除低置信样本{'未显著改善 (CI 重叠)' if overlap else '带来可见变化 (CI 不重叠)'}")
    print()

    # 文件输出
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_json = RESULTS_DIR / "svm_results_repeated.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"scheme_A": resA, "scheme_B": resB}, f, ensure_ascii=False, indent=2)
    out_png = RESULTS_DIR / "svm_confusion_matrix.png"
    plot_confusions(resA, resB, out_png)
    print(f"输出: {out_json}")
    print(f"输出: {out_png}")


if __name__ == "__main__":
    main()
