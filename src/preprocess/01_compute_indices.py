# -*- coding: utf-8 -*-
"""为每个 patch 计算光谱/SAR 指数 + salinity_prior 连续打分 (非最终标签)。

遍历 data/raw 下所有 patch_*.tif (60 个), 每个计算 6 个指数 + VV 均值, 综合成一个
基于 rank 的 salinity_prior 分数, 输出逐 patch 汇总到 data/labels/indices_summary.csv
(按 salinity_prior 降序), 并打印分布与 Top/Bottom 候选。

波段堆叠顺序 (rasterio 1-based): B2=1 B3=2 B4=3 B8=4 B11=5 B12=6 VV=7 VH=8

指数 (光学):
  1. NDVI      = (B8 - B4)/(B8 + B4)        植被
  2. NDWI      = (B3 - B8)/(B3 + B8)        水体
  3. SI        = sqrt(B2 * B4)              经典土壤盐分指数 (值域 0~数千, 高=盐高)
  4. NDSI-SWIR = (B11 - B12)/(B11 + B12)    SWIR 内部对比, 盐/干土
  5. SR-SWIR   = B11 / B12                  盐结皮 SWIR 比值
指数 (SAR):
  6. VH/VV     = VH / VV                    地表粗糙度 (盐碱异于农田/水体)
  (另算 VV 均值, 用于 salinity_prior 的 low_vv 分量)

salinity_prior: 对 5 个分量在 60 个 patch 内取 rank (favored 方向 -> 高 rank),
归一化 0-1 后加权求和; water_dominant 的 patch 乘以惩罚系数压低。

注意: 旧版 NDSI=(B4-B8)/(B4+B8) 恒等于 -NDVI (冗余), 已弃用。
最终标签仍以 QGIS 目视为准, 本脚本只给一个起点。

用法: conda run -n saline python src/preprocess/01_compute_indices.py
"""
# 决策记录 (2026-06-03): 已选定方案 B (连续 salinity_prior 打分), 取代旧的二分
#   likely_saline_candidate 启发式 (其 SR-SWIR>1.1 恒真、SI>median 未生效)。
#   v2 调整: low_ndvi 权重升至主导 (0.35), 并加植被惩罚 ×0.4 (仿水体 ×0.1),
#   解决"高植被污染 Top 榜"与"std 过小"两个问题。
#   (注: 直接用 rank 而非 rank/n 加权对 std 是无效的仿射变换, 已弃用。)
#   旧列 likely_saline_candidate 暂保留供对比, 后续可移除。
#   方案 C (引入土壤盐渍化 1:3200 万真值地图) 留待标注阶段叠加。
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

# --- 旧启发式阈值 (仅为保留 likely_saline_candidate 列) ---
SR_SWIR_ALT = 1.1

# --- salinity_prior 权重 (可调) ---
# low_ndvi 主导 (直接的"非植被"信号), SWIR 簇从 0.65 降到 0.50。
WEIGHTS = {
    "low_ndvi": 0.35,    # NDVI 低 (低植被) — 主导项
    "si": 0.20,          # SI 高 = 盐分高
    "ndsi_swir": 0.20,   # NDSI-SWIR 高
    "sr_swir": 0.10,     # SR-SWIR 高
    "low_vv": 0.15,      # VV 低 (后向散射低)
}
WATER_PENALTY = 0.1      # water_dominant 时乘此系数 (狠)
VEG_PENALTY = 0.4        # vegetation_dominant 时乘此系数 (宽容: 植被下可能仍有盐碱)


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
        "vv_mean": round(vmean(vv), 4),
        "water_dominant": water_frac > WATER_FRAC,
        "vegetation_dominant": veg_frac > VEG_FRAC,
        "bare_soil_dominant": bare_frac > BARE_FRAC,
    }


def compute_salinity_prior(df: pd.DataFrame) -> pd.DataFrame:
    """基于 rank 的加权 salinity_prior + 区内/总排名。"""
    n = len(df)
    # favored 方向 -> 高 rank (rank 60 最像盐碱), 归一化到 ~(0,1]
    norm_rank = {
        "si": df["si_mean"].rank(ascending=True) / n,           # SI 高
        "ndsi_swir": df["ndsi_swir_mean"].rank(ascending=True) / n,  # NDSI-SWIR 高
        "sr_swir": df["sr_swir_mean"].rank(ascending=True) / n,  # SR-SWIR 高
        "low_ndvi": df["ndvi_mean"].rank(ascending=False) / n,   # NDVI 低
        "low_vv": df["vv_mean"].rank(ascending=False) / n,       # VV 低
    }
    raw = sum(WEIGHTS[k] * norm_rank[k] for k in WEIGHTS)
    # 乘法惩罚: 植被 ×0.4, 水体 ×0.1 (近乎互斥, 同时命中则连乘)
    factor = np.ones(len(df))
    factor = np.where(df["vegetation_dominant"], factor * VEG_PENALTY, factor)
    factor = np.where(df["water_dominant"], factor * WATER_PENALTY, factor)
    df["salinity_prior"] = np.round(raw * factor, 4)

    df["rank_overall"] = (
        df["salinity_prior"].rank(ascending=False, method="first").astype(int)
    )
    df["rank_in_region"] = (
        df.groupby("region")["salinity_prior"]
        .rank(ascending=False, method="first")
        .astype(int)
    )
    return df.sort_values("salinity_prior", ascending=False).reset_index(drop=True)


def show(sub: pd.DataFrame) -> None:
    for _, r in sub.iterrows():
        print(
            f"  #{int(r['rank_overall']):2d} {r['patch_id']:20s} {r['region']:8s} "
            f"prior={r['salinity_prior']:.4f} | SI={r['si_mean']:7.1f} "
            f"NDSI-SWIR={r['ndsi_swir_mean']:6.3f} NDVI={r['ndvi_mean']:6.3f} "
            f"VV={r['vv_mean']:7.2f}"
        )


def main() -> None:
    tifs = sorted(DATA_RAW.glob("**/patch_*.tif"))
    print(f"找到 {len(tifs)} 个 patch")

    rows = [r for r in (process(p) for p in tifs) if r is not None]
    df = pd.DataFrame(rows)

    # 旧二分启发式 (保留列, 供对比)
    si_med = df["si_mean"].median()
    ndsi_med = df["ndsi_swir_mean"].median()
    df["likely_saline_candidate"] = (
        df["bare_soil_dominant"]
        & (df["si_mean"] > si_med)
        & ((df["ndsi_swir_mean"] > ndsi_med) | (df["sr_swir_mean"] > SR_SWIR_ALT))
        & (~df["water_dominant"])
    )

    # 方案 B: 连续打分
    df = compute_salinity_prior(df)

    # 输出列顺序
    cols = [
        "patch_id", "region", "salinity_prior", "rank_overall", "rank_in_region",
        "ndvi_mean", "ndwi_mean", "si_mean", "ndsi_swir_mean", "sr_swir_mean",
        "vh_vv_ratio_mean", "vv_mean",
        "water_dominant", "vegetation_dominant", "bare_soil_dominant",
        "likely_saline_candidate",
    ]
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df[cols].to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    # --- 控制台输出 ---
    sp = df["salinity_prior"]
    print("=" * 78)
    print(
        f"salinity_prior 分布: min={sp.min():.4f}  max={sp.max():.4f}  "
        f"mean={sp.mean():.4f}  median={sp.median():.4f}  std={sp.std():.4f}"
    )
    print("-" * 78)
    print("Top 10 候选 (总排名):")
    show(df.head(10))
    print("-" * 78)
    print("Bottom 10 (总排名):")
    show(df.tail(10))
    print("-" * 78)
    for reg in ("pingluo", "daan"):
        print(f"{reg} 区 Top 5:")
        show(df[df["region"] == reg].head(5))
    print("=" * 78)
    print("⚠️ salinity_prior 是基于指数排名的客观先验, 不是 ground truth。")
    print("   最终标签由 QGIS 目视判断决定。")
    print("   高 prior 不代表一定是盐碱, 低 prior 不代表一定不是。")
    print("   此分数仅用于引导标注顺序与边缘案例参考。")
    print(f"输出: {OUT_CSV}")


if __name__ == "__main__":
    main()
