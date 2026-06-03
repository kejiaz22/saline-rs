# -*- coding: utf-8 -*-
"""为每个 patch 计算光谱/SAR 指数, 给标注阶段提供客观参考 (非最终标签)。

遍历 data/raw 下所有 patch_*.tif (60 个), 每个计算 6 个指数, 输出逐 patch 汇总到
data/labels/indices_summary.csv, 并打印整体分布与盐碱候选。

波段堆叠顺序 (rasterio 1-based): B2=1 B3=2 B4=3 B8=4 B11=5 B12=6 VV=7 VH=8

指数 (光学):
  1. NDVI      = (B8 - B4)/(B8 + B4)        植被
  2. NDWI      = (B3 - B8)/(B3 + B8)        水体
  3. SI        = sqrt(B2 * B4)              经典土壤盐分指数 (值域 0~数千, 高=盐高)
  4. NDSI-SWIR = (B11 - B12)/(B11 + B12)    SWIR 内部对比, 盐/干土
  5. SR-SWIR   = B11 / B12                  盐结皮 SWIR 比值
指数 (SAR):
  6. VH/VV     = VH / VV                    地表粗糙度 (盐碱异于农田/水体)

注意: 旧版 NDSI=(B4-B8)/(B4+B8) 恒等于 -NDVI (冗余), 已弃用。
最终标签仍以 QGIS 目视为准, 本脚本只给一个起点。

用法: conda run -n saline python src/preprocess/01_compute_indices.py
"""
# TODO (Day 2): 当前的 likely_saline_candidate 启发式存在两个问题:
#   1. SR-SWIR > 1.1 是恒真条件 (实际最小值 1.135), 没有过滤作用
#   2. SI > median 实际未生效, 候选集退化为 bare_soil_dominant
#
# 明天讨论三个方案:
#   (A) 快修: AND 替代 OR, 让 SWIR 真正过滤
#   (B) 连续打分: salinity_prior 分数排序代替二分
#   (C) 引入真值: 用中国土壤盐渍化 1:3200 万地图做客观锚点
#
# 决策见 chat session 2026-06-03。
import sys
import re
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import rasterio

from config import DATA_RAW, DATA_LABELS

NODATA = -9999
OUT_CSV = DATA_LABELS / "indices_summary.csv"

# --- dominant 判定阈值 ---
WATER_NDWI_THRESH = 0.3    # NDWI > 此值算水体像元
WATER_FRAC = 0.5
VEG_NDVI_THRESH = 0.3      # NDVI > 此值算植被像元
VEG_FRAC = 0.5
BARE_NDVI_MAX = 0.2        # 裸土: NDVI < 此值
BARE_NDWI_MAX = 0.0        # 且 NDWI < 此值
BARE_FRAC = 0.3            # 裸土像元占比 > 此值 -> bare_soil_dominant

# --- 盐碱候选启发式 ---
SR_SWIR_ALT = 1.1          # SR-SWIR 备选阈值


def norm_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    denom = a + b
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(denom != 0, (a - b) / denom, np.nan)


def safe_ratio(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(b != 0, a / b, np.nan)


def region_of(patch_id: str) -> str:
    m = re.match(r"patch_([a-z]+)_\d+", patch_id)
    return m.group(1) if m else "?"


def process(tif_path: Path) -> dict:
    with rasterio.open(tif_path) as src:
        nd = src.nodata if src.nodata is not None else NODATA
        b2 = src.read(1).astype("float64")   # Blue
        b3 = src.read(2).astype("float64")   # Green
        b4 = src.read(3).astype("float64")   # Red
        b8 = src.read(4).astype("float64")   # NIR
        b11 = src.read(5).astype("float64")  # SWIR1
        b12 = src.read(6).astype("float64")  # SWIR2
        vv = src.read(7).astype("float64")   # SAR VV (dB)
        vh = src.read(8).astype("float64")   # SAR VH (dB)

    bands = [b2, b3, b4, b8, b11, b12, vv, vh]
    valid = np.ones_like(b2, dtype=bool)
    for arr in bands:
        valid &= (arr != nd) & ~np.isnan(arr)

    n_valid = int(valid.sum())
    if n_valid == 0:
        return None

    with np.errstate(invalid="ignore", divide="ignore"):
        ndvi = norm_diff(b8, b4)
        ndwi = norm_diff(b3, b8)
        si = np.sqrt(np.where(b2 * b4 >= 0, b2 * b4, np.nan))
        ndsi_swir = norm_diff(b11, b12)
        sr_swir = safe_ratio(b11, b12)
        vh_vv = safe_ratio(vh, vv)

    def vmean(arr):
        return float(np.nanmean(np.where(valid, arr, np.nan)))

    water_frac = float(np.sum((ndwi > WATER_NDWI_THRESH) & valid) / n_valid)
    veg_frac = float(np.sum((ndvi > VEG_NDVI_THRESH) & valid) / n_valid)
    bare_frac = float(
        np.sum((ndvi < BARE_NDVI_MAX) & (ndwi < BARE_NDWI_MAX) & valid) / n_valid
    )

    patch_id = tif_path.stem
    return {
        "patch_id": patch_id,
        "region": region_of(patch_id),
        "ndvi_mean": round(vmean(ndvi), 4),
        "ndwi_mean": round(vmean(ndwi), 4),
        "si_mean": round(vmean(si), 2),
        "ndsi_swir_mean": round(vmean(ndsi_swir), 4),
        "sr_swir_mean": round(vmean(sr_swir), 4),
        "vh_vv_ratio_mean": round(vmean(vh_vv), 4),
        "water_dominant": water_frac > WATER_FRAC,
        "vegetation_dominant": veg_frac > VEG_FRAC,
        "bare_soil_dominant": bare_frac > BARE_FRAC,
    }


def dist(s: pd.Series) -> str:
    return f"min={s.min():.3f}  max={s.max():.3f}  mean={s.mean():.3f}  median={s.median():.3f}"


def main() -> None:
    tifs = sorted(DATA_RAW.glob("**/patch_*.tif"))
    print(f"找到 {len(tifs)} 个 patch")

    rows = [r for r in (process(p) for p in tifs) if r is not None]
    df = pd.DataFrame(rows)

    # --- 盐碱候选 (依赖全体中位数, 第二遍) ---
    si_med = df["si_mean"].median()
    ndsi_med = df["ndsi_swir_mean"].median()
    df["likely_saline_candidate"] = (
        df["bare_soil_dominant"]
        & (df["si_mean"] > si_med)
        & ((df["ndsi_swir_mean"] > ndsi_med) | (df["sr_swir_mean"] > SR_SWIR_ALT))
        & (~df["water_dominant"])
    )

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    # --- 汇总 ---
    print("=" * 64)
    for col in (
        "ndvi_mean",
        "ndwi_mean",
        "si_mean",
        "ndsi_swir_mean",
        "sr_swir_mean",
        "vh_vv_ratio_mean",
    ):
        print(f"{col:18s}: {dist(df[col])}")
    print("-" * 64)
    print(f"water_dominant      : {int(df['water_dominant'].sum())} 个")
    print(f"vegetation_dominant : {int(df['vegetation_dominant'].sum())} 个")
    print(f"bare_soil_dominant  : {int(df['bare_soil_dominant'].sum())} 个")
    print("-" * 64)
    cand = df[df["likely_saline_candidate"]]
    by_region = cand.groupby("region").size().to_dict()
    print(
        f"likely_saline_candidate: {len(cand)} 个  "
        f"(pingluo {by_region.get('pingluo', 0)}, daan {by_region.get('daan', 0)})"
    )
    print(f"  启发式: bare_soil_dominant & SI>中位({si_med:.1f}) & "
          f"(NDSI-SWIR>中位({ndsi_med:.4f}) 或 SR-SWIR>{SR_SWIR_ALT}) & 非水体")
    if len(cand):
        print("  候选 patch_id:")
        for pid in cand["patch_id"].tolist():
            print(f"    {pid}")
    print(f"输出: {OUT_CSV}")


if __name__ == "__main__":
    main()
