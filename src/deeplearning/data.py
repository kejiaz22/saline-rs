# -*- coding: utf-8 -*-
"""PyTorch Dataset: 8 波段 patch -> (tensor[8,224,224], label)。

- 归一化: 用 features.csv 各波段的全局统计 (每波段 mean 取 *_mean 列均值,
  std 取 *_std 列均值, 即平均的波段内像元标准差, 比"均值的标准差"更适合归一化)。
- NoData(-9999) 替换为该波段全局均值 (归一化后即 ~0)。
- 尺寸统一到 224x224 (F.interpolate 双线性; 不用 cv2, 其对 8 通道支持差)。
- 训练增强: 随机水平/垂直翻转 + 随机 90° 倍数旋转 (遥感语义不变, 不引入空角)。
  (不做 ColorJitter, 会破坏波段物理意义。)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import rasterio
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

import config as cfg


def load_norm_stats() -> dict:
    """从 features.csv 计算每波段 (mean, std), 用于归一化。

    返回: {band_name: (mean, std)}。
    """
    df = pd.read_csv(cfg.DATA_PROCESSED / "features.csv")
    stats = {}
    for b in cfg.BAND_NAMES:
        mean = float(df[f"{b}_mean"].mean())
        std = float(df[f"{b}_std"].mean())
        stats[b] = (mean, std if std > 1e-6 else 1.0)
    return stats


def build_index() -> list:
    """读 labels_v1.csv, 构造 [(tif_path, label), ...] (只取 label∈{0,1})。"""
    labels = pd.read_csv(cfg.DATA_LABELS / "labels_v1.csv", dtype={"label": int})
    # patch_id -> tif 路径
    tif_map = {p.stem: p for p in cfg.DATA_RAW.glob("**/patch_*.tif")}
    items = []
    for _, r in labels.iterrows():
        if int(r["label"]) in (0, 1) and r["patch_id"] in tif_map:
            items.append((tif_map[r["patch_id"]], int(r["label"])))
    return items


class PatchDataset(Dataset):
    def __init__(self, items: list, norm_stats: dict, augment: bool = False):
        """items: [(tif_path, label), ...]; norm_stats: load_norm_stats() 结果。"""
        self.items = items
        self.norm = norm_stats
        self.augment = augment

    def __len__(self) -> int:
        return len(self.items)

    def _read_normalized(self, tif_path: Path) -> np.ndarray:
        """读 8 波段 -> NoData 填均值 -> 逐波段标准化 -> (8,H,W) float32。"""
        with rasterio.open(tif_path) as src:
            arr = src.read().astype("float32")  # (8, H, W)
            nd = src.nodata if src.nodata is not None else cfg.NODATA
        for i, b in enumerate(cfg.BAND_NAMES):
            mean, std = self.norm[b]
            band = arr[i]
            band[(band == nd) | np.isnan(band)] = mean  # NoData -> 均值
            arr[i] = (band - mean) / std
        return arr

    def _augment(self, x: torch.Tensor) -> torch.Tensor:
        """随机水平/垂直翻转 + 随机 90° 倍数旋转 (语义不变)。"""
        if torch.rand(1).item() < 0.5:
            x = torch.flip(x, dims=[-1])          # 水平翻转
        if torch.rand(1).item() < 0.5:
            x = torch.flip(x, dims=[-2])          # 垂直翻转
        k = int(torch.randint(0, 4, (1,)).item())  # 0/90/180/270 度
        if k:
            x = torch.rot90(x, k, dims=[-2, -1])
        return x

    def __getitem__(self, idx: int):
        tif_path, label = self.items[idx]
        arr = self._read_normalized(tif_path)
        x = torch.from_numpy(arr)                 # (8, H, W)
        # 统一到 224x224
        x = F.interpolate(
            x.unsqueeze(0), size=(cfg.INPUT_SIZE, cfg.INPUT_SIZE),
            mode="bilinear", align_corners=False,
        ).squeeze(0)
        if self.augment:
            x = self._augment(x)
        return x, label
