# -*- coding: utf-8 -*-
"""Random Forest 训练 + feature importance, 与 SVM 对比 (5x10 repeated CV)。

输入:
  - data/processed/features.csv
  - results/svm_results_repeated.json  (升级版 SVM 结果, 用于对比)

# 环境注记: 本 Windows conda 环境下, sklearn 森林的 OpenMP 在同一进程内大量 fit
# (50 折 nested GridSearch ≈ 7200 次) 后必然 access violation 崩溃 (0xC0000005),
# 即使限制线程 / 裁剪网格亦然 (~1800 次仍崩)。_debug 验证 ~150 次以内稳定。
# 故 RF 改为「一次性选超参 (GridSearch, cv=3) -> 固定超参做 5x10 repeated CV」,
# 总 fit 数 ≈ 136, 稳定。代价: 超参选择看了全量数据 (轻微乐观), A/B 共用超参
# (反而使消融成为只变数据的受控对比)。SVM 仍用 nested CV (SVC 不触发该崩溃)。

CV: RepeatedStratifiedKFold(5 x 10 = 50)。
RF 不需标准化, 但保持 Pipeline 与 SVM 一致便于对比。

独家产物: 50 fold 的 feature_importances_ 取 mean±std, 出 Top 15 + 与 Spearman 对照。

输出:
  - results/rf_results.json
  - results/rf_feature_importance.png
  - results/svm_vs_rf_confusion.png
  - results/svm_vs_rf_comparison.csv

用法: conda run -n saline python src/classical/03_train_rf.py
"""
import os
# 必须在 import numpy/sklearn 之前: 限制线程 + 允许重复 OMP 运行时, 缓解森林崩溃。
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

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
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedStratifiedKFold, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
)

from config import DATA_PROCESSED, RESULTS_DIR, RANDOM_SEED

FEATURES_CSV = DATA_PROCESSED / "features.csv"
SVM_JSON = RESULTS_DIR / "svm_results_repeated.json"
META_COLS = ["patch_id", "region", "label", "confidence"]

PARAM_GRID = {
    "rf__n_estimators": [100, 200, 500],
    "rf__max_depth": [None, 10],
    "rf__min_samples_leaf": [1, 2],
}
N_SPLITS, N_REPEATS, SELECT_CV = 5, 10, 3


def build_pipeline() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),  # RF 不需要, 但与 SVM 对齐
        ("rf", RandomForestClassifier(random_state=RANDOM_SEED, n_jobs=1)),
    ])


def ci95(std: float, n: int) -> float:
    return 1.96 * std / np.sqrt(n)


def select_hyperparams(X: np.ndarray, y: np.ndarray) -> dict:
    """一次性 GridSearch(cv=3) 选超参 (~36 fit)。"""
    grid = GridSearchCV(build_pipeline(), PARAM_GRID, scoring="f1",
                        cv=SELECT_CV, n_jobs=1)
    grid.fit(X, y)
    return grid.best_params_


def evaluate_fixed(X: np.ndarray, y: np.ndarray, params: dict, feat_cols: list) -> dict:
    """固定超参做 5x10 repeated CV (~50 fit), 收集指标/重要性/混淆矩阵。"""
    skf = RepeatedStratifiedKFold(
        n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=RANDOM_SEED
    )
    fold_metrics, importances = [], []
    cm_total = np.zeros((2, 2), dtype=int)

    for tr, va in skf.split(X, y):
        pipe = build_pipeline().set_params(**params)
        pipe.fit(X[tr], y[tr])
        pred = pipe.predict(X[va])
        yt = y[va]
        fold_metrics.append({
            "accuracy": accuracy_score(yt, pred),
            "precision": precision_score(yt, pred, pos_label=1, zero_division=0),
            "recall": recall_score(yt, pred, pos_label=1, zero_division=0),
            "f1": f1_score(yt, pred, pos_label=1, zero_division=0),
            "f1_macro": f1_score(yt, pred, average="macro", zero_division=0),
        })
        importances.append(pipe.named_steps["rf"].feature_importances_)
        cm_total += confusion_matrix(yt, pred, labels=[0, 1])

    n_est = len(fold_metrics)
    agg = {}
    for k in fold_metrics[0]:
        vals = np.array([m[k] for m in fold_metrics])
        std = float(np.std(vals, ddof=1))
        agg[k] = {"mean": float(np.mean(vals)), "std": std, "ci95": ci95(std, n_est)}

    imp = np.array(importances)
    imp_df = pd.DataFrame({
        "feature": feat_cols,
        "imp_mean": imp.mean(axis=0),
        "imp_std": imp.std(axis=0, ddof=1),
    }).sort_values("imp_mean", ascending=False).reset_index(drop=True)

    return {
        "n": int(len(y)),
        "params": {k.replace("rf__", ""): params[k] for k in params},
        "aggregate": agg,
        "confusion_total": cm_total.tolist(),
        "importance": imp_df.to_dict(orient="records"),
    }


def spearman_vs_label(df: pd.DataFrame, feat_cols: list) -> list:
    y_rank = df["label"].rank().values
    out = []
    for c in feat_cols:
        r = np.corrcoef(df[c].rank().values, y_rank)[0, 1]
        out.append((c, float(r)))
    out.sort(key=lambda t: abs(t[1]), reverse=True)
    return out


def print_scheme(title: str, res: dict) -> None:
    a = res["aggregate"]
    print(f"【{title} (n={res['n']})】")
    print(f"  固定超参: {res['params']}")
    for key, label in [("accuracy", "Accuracy"), ("f1", "F1 (saline)"),
                       ("f1_macro", "F1 macro"), ("precision", "Precision(1)"),
                       ("recall", "Recall(1)")]:
        m = a[key]
        print(f"  {label:13s}: {m['mean']:.3f} ± {m['std']:.3f}  "
              f"(95% CI: [{m['mean']-m['ci95']:.3f}, {m['mean']+m['ci95']:.3f}])")
    cm = res["confusion_total"]
    print("  Confusion matrix (50-fold 累加):")
    print("                 Pred 0  Pred 1")
    print(f"        True 0:   {cm[0][0]:5d}   {cm[0][1]:5d}")
    print(f"        True 1:   {cm[1][0]:5d}   {cm[1][1]:5d}")
    print()


def main() -> None:
    df = pd.read_csv(FEATURES_CSV)
    feat_cols = [c for c in df.columns if c not in META_COLS]

    Xa, ya = df[feat_cols].values, df["label"].values.astype(int)
    dfB = df[df["confidence"].isin(["high", "medium"])].reset_index(drop=True)
    Xb, yb = dfB[feat_cols].values, dfB["label"].values.astype(int)

    print("=" * 70)
    print("Random Forest 训练结果 (一次性选超参 + 5x10 RepeatedStratifiedKFold)")
    print("=" * 70 + "\n")

    # 一次性选超参 (在全样本 A 上), A/B 共用
    best = select_hyperparams(Xa, ya)
    print(f"选定超参 (GridSearch cv=3 @ 全样本): "
          f"{{ {', '.join(k.replace('rf__','')+'='+str(v) for k,v in best.items())} }}\n")

    resA = evaluate_fixed(Xa, ya, best, feat_cols)
    print_scheme("方案 A: 全样本", resA)
    resB = evaluate_fixed(Xb, yb, best, feat_cols)
    print_scheme("方案 B: 剔除 low-confidence", resB)

    fa, fb = resA["aggregate"]["f1"]["mean"], resB["aggregate"]["f1"]["mean"]
    print(f"【RF 消融对比】 A F1={fa:.3f}  B F1={fb:.3f}  差异={fb-fa:+.3f}")
    print("  -> RF 是否也出现 SVM 那样的反直觉下降?", "是" if fb < fa else "否")
    print()

    # Feature importance Top 15
    imp_df = pd.DataFrame(resA["importance"])
    print("=" * 70)
    print("Feature Importance (Top 15, mean over 50 folds)")
    print("=" * 70)
    print(f"{'Rank':<5}{'Feature':<18}{'Importance (mean ± std)'}")
    for i, row in imp_df.head(15).iterrows():
        print(f"{i+1:<5}{row['feature']:<18}{row['imp_mean']:.4f} ± {row['imp_std']:.4f}")
    print()

    # Spearman Top5 vs RF importance
    sp = spearman_vs_label(df, feat_cols)
    imp_rank = {r["feature"]: (i + 1, r["imp_mean"]) for i, r in enumerate(resA["importance"])}
    print("=" * 70)
    print("Spearman Top 5 vs RF Importance 对照")
    print("=" * 70)
    print(f"{'Spearman排名':<12}{'特征':<18}{'rho':<9}{'RF importance':<15}{'RF排名'}")
    for i, (feat, rho) in enumerate(sp[:5], start=1):
        rk, imp = imp_rank[feat]
        print(f"{i:<12}{feat:<18}{rho:+.3f}   {imp:.4f}        {rk}")
    print()

    # SVM vs RF 对比
    with open(SVM_JSON, encoding="utf-8") as f:
        svm = json.load(f)
    svmA = svm["scheme_A"]["aggregate"]
    rfA = resA["aggregate"]
    print("=" * 70)
    print("SVM vs RF 对比 (全样本, 5x10 RepeatedStratifiedKFold)")
    print("=" * 70)
    print(f"{'指标':<14}{'SVM mean±std':<18}{'[95% CI]':<20}{'RF mean±std':<18}{'[95% CI]'}")
    rows = []
    for key, label in [("accuracy", "Accuracy"), ("f1", "F1 (saline)"),
                       ("f1_macro", "F1 macro"), ("precision", "Precision(1)"),
                       ("recall", "Recall(1)")]:
        s, r = svmA[key], rfA[key]
        sci = f"[{s['mean']-s['ci95']:.3f},{s['mean']+s['ci95']:.3f}]"
        rci = f"[{r['mean']-r['ci95']:.3f},{r['mean']+r['ci95']:.3f}]"
        print(f"{label:<14}{s['mean']:.3f}±{s['std']:.3f}     {sci:<20}"
              f"{r['mean']:.3f}±{r['std']:.3f}     {rci}")
        rows.append({
            "metric": label,
            "svm_mean": round(s["mean"], 4), "svm_std": round(s["std"], 4),
            "svm_ci_low": round(s["mean"] - s["ci95"], 4), "svm_ci_high": round(s["mean"] + s["ci95"], 4),
            "rf_mean": round(r["mean"], 4), "rf_std": round(r["std"], 4),
            "rf_ci_low": round(r["mean"] - r["ci95"], 4), "rf_ci_high": round(r["mean"] + r["ci95"], 4),
        })
    sf, rf = svmA["f1"], rfA["f1"]
    overlap = not (rf["mean"] - rf["ci95"] > sf["mean"] + sf["ci95"]
                   or sf["mean"] - sf["ci95"] > rf["mean"] + rf["ci95"])
    print(f"\n总结: F1(saline) 两者 95% CI {'重叠 (无法区分)' if overlap else '不重叠 (有显著差异)'}; "
          f"{'RF' if rf['mean']>sf['mean'] else 'SVM'} 点估计更高, "
          f"{'RF' if rf['std']<sf['std'] else 'SVM'} 更稳 (std 更小)")

    # 文件输出
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "rf_results.json", "w", encoding="utf-8") as f:
        json.dump({"scheme_A": resA, "scheme_B": resB, "selected_params": best}, f,
                  ensure_ascii=False, indent=2)
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "svm_vs_rf_comparison.csv", index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(8, 6))
    top = imp_df.head(15).iloc[::-1]
    ax.barh(top["feature"], top["imp_mean"], xerr=top["imp_std"], color="seagreen")
    ax.set_title("RF Feature Importance (Top 15, mean±std over 50 folds)")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "rf_feature_importance.png", dpi=120)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, cm, title, f1m in [
        (axes[0], np.array(svm["scheme_A"]["confusion_total"]), "SVM (A)", sf["mean"]),
        (axes[1], np.array(resA["confusion_total"]), "RF (A)", rf["mean"]),
    ]:
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                    xticklabels=["Pred 0", "Pred 1"], yticklabels=["True 0", "True 1"])
        ax.set_title(f"{title}  F1(saline)={f1m:.3f}")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "svm_vs_rf_confusion.png", dpi=120)
    plt.close(fig)

    print(f"\n输出: {RESULTS_DIR / 'rf_results.json'}")
    print(f"输出: {RESULTS_DIR / 'rf_feature_importance.png'}")
    print(f"输出: {RESULTS_DIR / 'svm_vs_rf_confusion.png'}")
    print(f"输出: {RESULTS_DIR / 'svm_vs_rf_comparison.csv'}")


if __name__ == "__main__":
    main()
