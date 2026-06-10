# -*- coding: utf-8 -*-
"""EfficientViT-M2 (8 通道) 训练: 复用 ResNet50 的训练框架, 只换模型。

与 train.py 完全相同的设置 (LR/patience/class weights/augmentation/5x3 RepeatedSKF),
仅模型换成 build_efficientvit_8channel, 结果输出到 efficientvit_results.json。
公平对比 ResNet50 用。

用法 (Colab): python -m src.deeplearning.train_efficientvit
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
from sklearn.model_selection import RepeatedStratifiedKFold

import config as cfg
from data import PatchDataset, load_norm_stats, build_index
from models import build_efficientvit_8channel
from train import set_seed, evaluate, ci95  # 复用无状态辅助函数

EVIT_RESULTS_JSON = cfg.RESULTS_DIR / "efficientvit_results.json"


def train_one_fold(items: list, train_idx, val_idx, fold_num: int,
                   norm_stats: dict) -> dict:
    """与 train.py.train_one_fold 一致, 仅模型换成 EfficientViT-M2。"""
    train_items = [items[i] for i in train_idx]
    val_items = [items[i] for i in val_idx]
    train_ds = PatchDataset(train_items, norm_stats, augment=True)
    val_ds = PatchDataset(val_items, norm_stats, augment=False)
    train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.BATCH_SIZE, shuffle=False)

    model = build_efficientvit_8channel().to(cfg.DEVICE)
    optimizer = AdamW(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.T_MAX)
    # 同 ResNet: 类别权重补偿 21:39 不平衡
    class_weights = torch.tensor([1.0, 39.0 / 21.0]).to(cfg.DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    cfg.CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = cfg.CHECKPOINTS_DIR / f"efficientvit_fold{fold_num}.pt"

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
        if (epoch >= cfg.MIN_EPOCHS_BEFORE_EARLY_STOP
                and no_improve >= cfg.EARLY_STOP_PATIENCE):
            print(f"  fold{fold_num} early stop @ epoch {epoch}")
            break

    return best_metrics


def main() -> None:
    set_seed(cfg.RANDOM_SEED)
    print(f"device: {cfg.DEVICE}")

    items = build_index()
    norm_stats = load_norm_stats()
    y_all = np.array([lbl for _, lbl in items])
    print(f"samples: {len(items)}  (saline={int((y_all==1).sum())}, "
          f"non={int((y_all==0).sum())})")

    skf = RepeatedStratifiedKFold(n_splits=cfg.N_SPLITS, n_repeats=cfg.N_REPEATS,
                                  random_state=cfg.RANDOM_SEED)
    fold_results = []
    for fold, (tr, va) in enumerate(skf.split(np.zeros(len(items)), y_all)):
        print(f"=== Fold {fold} (train={len(tr)} val={len(va)}) ===")
        fold_results.append(train_one_fold(items, tr, va, fold, norm_stats))

    agg = {}
    for k in ("accuracy", "precision", "recall", "f1", "f1_macro"):
        vals = np.array([r[k] for r in fold_results])
        std = float(np.std(vals, ddof=1))
        agg[k] = {"mean": float(vals.mean()), "std": std,
                  "ci95": ci95(std, len(fold_results))}

    print("\n" + "=" * 60)
    print(f"EfficientViT-M2 (8ch) {cfg.N_SPLITS}x{cfg.N_REPEATS} RepeatedSKF "
          f"({len(fold_results)} folds) 结果")
    print("=" * 60)
    for k in ("accuracy", "f1", "f1_macro", "precision", "recall"):
        m = agg[k]
        print(f"  {k:10s}: {m['mean']:.3f} ± {m['std']:.3f}  "
              f"(95% CI: [{m['mean']-m['ci95']:.3f}, {m['mean']+m['ci95']:.3f}])")

    # 与基线 + ResNet50 对比
    print("-" * 60)
    refs = [("SVM", cfg.SVM_JSON, "scheme_A"), ("RF", cfg.RF_JSON, "scheme_A"),
            ("ResNet50", cfg.RESULTS_JSON, None)]
    for name, jp, key in refs:
        if jp.exists():
            d = json.load(open(jp, encoding="utf-8"))
            f1 = (d[key]["aggregate"]["f1"] if key else d["aggregate"]["f1"])["mean"]
            print(f"  {name:10s} F1(saline): {f1:.3f}")
    print(f"  EfficientViT F1(saline): {agg['f1']['mean']:.3f}  (目标 ≥ 0.65)")

    cfg.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(EVIT_RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump({"aggregate": agg, "folds": fold_results}, f,
                  ensure_ascii=False, indent=2)
    print(f"\n输出: {EVIT_RESULTS_JSON}")


if __name__ == "__main__":
    main()
