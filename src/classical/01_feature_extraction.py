# -*- coding: utf-8 -*-
"""特征工程: 为每个 patch 提取 ~34 个特征, 供传统分类器 (SVM/RF) 使用。

输入:
  - data/raw 下 patch_*.tif (60 个)
  - data/labels/labels_v1.csv  (label, confidence)
  - data/labels/indices_summary.csv  (6 个衍生指数)

特征 (共 34):
  A. 光谱统计 (24): 6 光学波段 B2/B3/B4/B8/B11/B12 各 mean/std/p25/p75
  B. SAR 统计 (4) : VV mean/std, VH mean/std
  C. 衍生指数 (6) : ndvi/ndwi/si/ndsi_swir/sr_swir/vh_vv_ratio (从 indices_summary join)

输出: data/processed/features.csv  (60 行 x 38 列 = 4 元数据 + 34 特征)

NoData(-9999) 在统计前 mask 掉。

用法: conda run -n saline python src/classical/01_feature_extraction.py
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import rasterio

from config import DATA_RAW, DATA_LABELS, DATA_PROCESSED

NODATA = -9999
OUT_CSV = DATA_PROCESSED / "features.csv"

# rasterio 1-based 波段索引 -> 名称
OPTICAL = [(1, "B2"), (2, "B3"), (3, "B4"), (4, "B8"), (5, "B11"), (6, "B12")]
SAR = [(7, "VV"), (8, "VH")]
INDEX_COLS = [
    "ndvi_mean", "ndwi_mean", "si_mean",
    "ndsi_swir_mean", "sr_swir_mean", "vh_vv_ratio_mean",
]


def band_stats(arr: np.ndarray, valid: np.ndarray, name: str, full: bool) -> dict:
    """单波段统计。full=True 出 mean/std/p25/p75, 否则只 mean/std。"""
    v = arr[valid]
    out = {f"{name}_mean": float(np.mean(v)), f"{name}_std": float(np.std(v))}
    if full:
        out[f"{name}_p25"] = float(np.percentile(v, 25))
        out[f"{name}_p75"] = float(np.percentile(v, 75))
    return out


def extract(tif_path: Path) -> dict:
    with rasterio.open(tif_path) as src:
        nd = src.nodata if src.nodata is not None else NODATA
        bands = {idx: src.read(idx).astype("float64") for idx, _ in OPTICAL + SAR}

    valid = np.ones_like(next(iter(bands.values())), dtype=bool)
    for arr in bands.values():
        valid &= (arr != nd) & ~np.isnan(arr)

    feats = {"patch_id": tif_path.stem}
    for idx, name in OPTICAL:
        feats.update(band_stats(bands[idx], valid, name, full=True))
    for idx, name in SAR:
        feats.update(band_stats(bands[idx], valid, name, full=False))
    return feats


def main() -> None:
    tifs = sorted(DATA_RAW.glob("**/patch_*.tif"))
    print(f"找到 {len(tifs)} 个 patch, 提取特征...")
    spectral = pd.DataFrame([extract(p) for p in tifs])

    # join labels + indices
    labels = pd.read_csv(DATA_LABELS / "labels_v1.csv", dtype={"label": str})
    labels = labels[["patch_id", "label", "confidence"]]
    indices = pd.read_csv(DATA_LABELS / "indices_summary.csv")
    indices = indices[["patch_id", "region"] + INDEX_COLS]

    df = spectral.merge(labels, on="patch_id").merge(indices, on="patch_id")
    df["label"] = df["label"].astype(int)

    # 列顺序
    spectral_cols = []
    for _, name in OPTICAL:
        spectral_cols += [f"{name}_mean", f"{name}_std", f"{name}_p25", f"{name}_p75"]
    sar_cols = ["VV_mean", "VV_std", "VH_mean", "VH_std"]
    cols = ["patch_id", "region", "label", "confidence"] + spectral_cols + sar_cols + INDEX_COLS
    df = df[cols]

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    # ---------- 验证 ----------
    feat_cols = spectral_cols + sar_cols + INDEX_COLS
    print("=" * 70)
    print(f"1. features.csv shape: {df.shape}  (期望 60 x 38)")
    print(f"   元数据 4 + 特征 {len(feat_cols)} = {4 + len(feat_cols)} 列")
    print("-" * 70)

    print("2. 各特征 min/max:")
    for c in feat_cols:
        print(f"   {c:18s}: min={df[c].min():12.4f}  max={df[c].max():12.4f}")
    print("-" * 70)

    nan_cells = df[feat_cols].isna().sum().sum()
    print(f"3. NaN 数量: {int(nan_cells)}")
    if nan_cells:
        for c in feat_cols:
            bad = df[df[c].isna()]["patch_id"].tolist()
            if bad:
                print(f"   !! 列 {c} NaN -> {bad}")
    print("-" * 70)

    print("4. label 分布:")
    print(f"   saline(1)={int((df['label']==1).sum())}  "
          f"non-saline(0)={int((df['label']==0).sum())}  "
          f"skip(-1)={int((df['label']==-1).sum())}")
    print("-" * 70)

    # 5. Spearman 相关 (秩相关 = 对 rank 求 Pearson)
    print("5. 各特征与 label 的 |Spearman| 相关 Top 5:")
    y_rank = pd.Series(df["label"]).rank().values
    corrs = []
    for c in feat_cols:
        x_rank = df[c].rank().values
        r = np.corrcoef(x_rank, y_rank)[0, 1]
        corrs.append((c, r))
    corrs.sort(key=lambda t: abs(t[1]), reverse=True)
    for c, r in corrs[:5]:
        print(f"   {c:18s}: rho={r:+.3f}")

    print(f"输出: {OUT_CSV}")


if __name__ == "__main__":
    main()
