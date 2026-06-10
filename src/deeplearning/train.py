# -*- coding: utf-8 -*-
"""ResNet50 (8 通道) 训练: 5-fold CV + 早停 + 与传统基线对比。

工作流: 本地写 -> push GitHub -> Colab (T4) clone 跑 -> 结果回写。

注意:
  - 训练数据 (60 个 tif) 与 features.csv 均不在 git (.gitignore),
    Colab 上需另行准备 (挂载 Google Drive 或上传); 详见 README/周报。
  - 首次运行会下载 ImageNet 预训练权重 (~100MB, 需联网)。

用法 (Colab): python src/deeplearning/train.py
"""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
)

import config as cfg
from data import PatchDataset, load_norm_stats, build_index
from models import build_resnet50_8channel


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def evaluate(model: nn.Module, loader: DataLoader) -> dict:
    """在 loader 上评估, 返回指标 + 预测/真值。"""
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(cfg.DEVICE)
            logits = model(x)
            pred = logits.argmax(dim=1).cpu().numpy()
            ps.extend(pred.tolist())
            ys.extend(y.numpy().tolist())
    ys, ps = np.array(ys), np.array(ps)
    return {
        "accuracy": accuracy_score(ys, ps),
        "precision": precision_score(ys, ps, pos_label=1, zero_division=0),
        "recall": recall_score(ys, ps, pos_label=1, zero_division=0),
        "f1": f1_score(ys, ps, pos_label=1, zero_division=0),
        "f1_macro": f1_score(ys, ps, average="macro", zero_division=0),
        "confusion": confusion_matrix(ys, ps, labels=[0, 1]).tolist(),
    }


def train_one_fold(items: list, train_idx, val_idx, fold_num: int,
                   norm_stats: dict) -> dict:
    """训练单个 fold, 返回该 fold 最佳 (val F1) 时的指标。"""
    train_items = [items[i] for i in train_idx]
    val_items = [items[i] for i in val_idx]
    train_ds = PatchDataset(train_items, norm_stats, augment=True)
    val_ds = PatchDataset(val_items, norm_stats, augment=False)
    train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.BATCH_SIZE, shuffle=False)

    model = build_resnet50_8channel().to(cfg.DEVICE)
    optimizer = AdamW(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.T_MAX)
    criterion = nn.CrossEntropyLoss()

    cfg.CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = cfg.CHECKPOINTS_DIR / f"resnet50_fold{fold_num}.pt"

    best_f1 = -1.0
    best_metrics = None
    no_improve = 0

    for epoch in range(cfg.EPOCHS):
        model.train()
        for x, y in train_loader:
            x, y = x.to(cfg.DEVICE), y.to(cfg.DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
        scheduler.step()

        metrics = evaluate(model, val_loader)
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_metrics = {**metrics, "best_epoch": epoch}
            torch.save(model.state_dict(), ckpt_path)
            no_improve = 0
        else:
            no_improve += 1

        print(f"  fold{fold_num} epoch{epoch:02d} "
              f"val_f1={metrics['f1']:.3f} best={best_f1:.3f} "
              f"(no_improve={no_improve})")
        if no_improve >= cfg.EARLY_STOP_PATIENCE:
            print(f"  fold{fold_num} early stop @ epoch {epoch}")
            break

    return best_metrics


def ci95(std: float, n: int) -> float:
    return 1.96 * std / np.sqrt(n)


def main() -> None:
    set_seed(cfg.RANDOM_SEED)
    print(f"device: {cfg.DEVICE}")

    items = build_index()
    norm_stats = load_norm_stats()
    y_all = np.array([lbl for _, lbl in items])
    print(f"samples: {len(items)}  (saline={int((y_all==1).sum())}, "
          f"non={int((y_all==0).sum())})")

    skf = StratifiedKFold(n_splits=cfg.N_SPLITS, shuffle=True,
                          random_state=cfg.RANDOM_SEED)
    fold_results = []
    for fold, (tr, va) in enumerate(skf.split(np.zeros(len(items)), y_all)):
        print(f"=== Fold {fold} (train={len(tr)} val={len(va)}) ===")
        fold_results.append(train_one_fold(items, tr, va, fold, norm_stats))

    # 聚合
    agg = {}
    for k in ("accuracy", "precision", "recall", "f1", "f1_macro"):
        vals = np.array([r[k] for r in fold_results])
        std = float(np.std(vals, ddof=1))
        agg[k] = {"mean": float(vals.mean()), "std": std,
                  "ci95": ci95(std, cfg.N_SPLITS)}

    print("\n" + "=" * 60)
    print("ResNet50 (8ch) 5-fold 结果")
    print("=" * 60)
    for k in ("accuracy", "f1", "f1_macro", "precision", "recall"):
        m = agg[k]
        print(f"  {k:10s}: {m['mean']:.3f} ± {m['std']:.3f}  "
              f"(95% CI: [{m['mean']-m['ci95']:.3f}, {m['mean']+m['ci95']:.3f}])")

    # 与传统基线对比
    print("-" * 60)
    for name, jp in [("SVM", cfg.SVM_JSON), ("RF", cfg.RF_JSON)]:
        if jp.exists():
            base = json.load(open(jp, encoding="utf-8"))["scheme_A"]["aggregate"]["f1"]
            print(f"  baseline {name} F1(saline): {base['mean']:.3f}")
    print(f"  ResNet50 F1(saline)       : {agg['f1']['mean']:.3f}  "
          f"(目标 ≥ 0.65)")

    # 保存
    cfg.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(cfg.RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump({"aggregate": agg, "folds": fold_results}, f,
                  ensure_ascii=False, indent=2)
    print(f"\n输出: {cfg.RESULTS_JSON}")


if __name__ == "__main__":
    main()
