# -*- coding: utf-8 -*-
"""一次性预处理: 为每个 patch 渲染两张 512x512 PNG 供网页标注用。

对 data/raw 下每个 patch_*.tif 生成:
  1. RGB 真彩色 : B4/B3/B2 -> R/G/B  -> {patch_id}_rgb.png
  2. SWIR 假彩色: B11/B8/B4 -> R/G/B -> {patch_id}_swir.png

每个波段按有效像元的 2-98 百分位拉伸; NoData(-9999) 显示为黑色;
从原始 ~402x402 上采样到 512x512。

输出目录: data/labels/preview_images/

用法: conda run -n saline python src/labeling/preprocess_preview.py
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import numpy as np
import rasterio
from PIL import Image

from config import DATA_RAW, DATA_LABELS

NODATA = -9999
OUT_DIR = DATA_LABELS / "preview_images"
OUT_SIZE = 512

# rasterio 1-based 波段索引
RGB_BANDS = [3, 2, 1]    # B4, B3, B2
SWIR_BANDS = [5, 4, 3]   # B11, B8, B4


def stretch(band: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """按有效像元 2-98 百分位线性拉伸到 [0,1]。"""
    vals = band[valid]
    if vals.size == 0:
        return np.zeros_like(band)
    lo, hi = np.percentile(vals, [2, 98])
    if hi <= lo:
        hi = lo + 1.0
    return np.clip((band - lo) / (hi - lo), 0.0, 1.0)


def render(tif_path: Path, band_idx: list, out_path: Path) -> None:
    with rasterio.open(tif_path) as src:
        nd = src.nodata if src.nodata is not None else NODATA
        arrs = [src.read(i).astype("float64") for i in band_idx]

    valid = np.ones_like(arrs[0], dtype=bool)
    for a in arrs:
        valid &= (a != nd) & ~np.isnan(a)

    h, w = arrs[0].shape
    rgb = np.zeros((h, w, 3), dtype=np.float64)
    for c, a in enumerate(arrs):
        rgb[..., c] = stretch(a, valid)
    rgb[~valid] = 0.0  # NoData -> 黑

    img = Image.fromarray((rgb * 255).astype("uint8"), mode="RGB")
    img = img.resize((OUT_SIZE, OUT_SIZE), Image.Resampling.LANCZOS)
    img.save(out_path)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tifs = sorted(DATA_RAW.glob("**/patch_*.tif"))
    print(f"找到 {len(tifs)} 个 patch, 渲染中...")

    count = 0
    for p in tifs:
        pid = p.stem
        render(p, RGB_BANDS, OUT_DIR / f"{pid}_rgb.png")
        render(p, SWIR_BANDS, OUT_DIR / f"{pid}_swir.png")
        count += 2

    print(f"完成: 生成 {count} 张 PNG -> {OUT_DIR}")


if __name__ == "__main__":
    main()
