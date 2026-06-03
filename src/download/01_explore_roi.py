# -*- coding: utf-8 -*-
"""探索两个研究区的影像可用性 (探索性, 不导出任何数据)。

对黄河中上游 (银川) 和东北松嫩 (大安) 两个研究区:
  - 用 config.py 的中心坐标各扩 0.5 度构造 bounding box
  - 筛选 2023 生长季 (6-9 月) 的 Sentinel-2 SR (云量<20%) 与 Sentinel-1 GRD (VV+VH, IW)
  - 只打印影像数量与首景信息, 不做任何 export

用法: conda run -n saline python src/download/01_explore_roi.py
"""
import sys
from pathlib import Path

# 把项目根目录加入 sys.path 以便 import config
sys.path.append(str(Path(__file__).resolve().parents[2]))

import ee
from config import GEE_PROJECT_ID, STUDY_AREAS, DATE_START, DATE_END

# bounding box 半边长 (度)
HALF_DEG = 0.5
# S2 云量上限 (%)
MAX_CLOUD = 20


def build_bbox(lat: float, lon: float) -> ee.Geometry:
    """以 (lon, lat) 为中心, 各扩 HALF_DEG 度构造矩形。"""
    return ee.Geometry.Rectangle(
        [lon - HALF_DEG, lat - HALF_DEG, lon + HALF_DEG, lat + HALF_DEG]
    )


def get_s2(bbox: ee.Geometry) -> ee.ImageCollection:
    """Sentinel-2 SR HARMONIZED, 生长季, 云量 < MAX_CLOUD。"""
    return (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterDate(DATE_START, DATE_END)
        .filterBounds(bbox)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", MAX_CLOUD))
        .sort("system:time_start")
    )


def get_s1(bbox: ee.Geometry) -> ee.ImageCollection:
    """Sentinel-1 GRD, IW 模式, VV+VH 双极化, 生长季。"""
    return (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterDate(DATE_START, DATE_END)
        .filterBounds(bbox)
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
        .sort("system:time_start")
    )


def fmt_date(img: ee.Image) -> str:
    """取影像采集日期, 格式 YYYY-MM-DD。"""
    return img.date().format("YYYY-MM-dd").getInfo()


def explore(key: str, area: dict) -> None:
    """打印单个研究区的影像可用性。"""
    bbox = build_bbox(area["lat"], area["lon"])
    s2 = get_s2(bbox)
    s1 = get_s1(bbox)

    print(f"=== {area['name']} ===")
    n_s2 = s2.size().getInfo()
    n_s1 = s1.size().getInfo()
    print(f"S2 影像数: {n_s2}")
    print(f"S1 影像数: {n_s1}")

    if n_s2 > 0:
        first_s2 = ee.Image(s2.first())
        cloud = first_s2.get("CLOUDY_PIXEL_PERCENTAGE").getInfo()
        print(f"S2 首景: {fmt_date(first_s2)}, 云量 {cloud:.1f}%")
    else:
        print("S2 首景: (无符合条件影像)")

    if n_s1 > 0:
        first_s1 = ee.Image(s1.first())
        print(f"S1 首景: {fmt_date(first_s1)}")
    else:
        print("S1 首景: (无符合条件影像)")
    print()


def main() -> None:
    ee.Initialize(project=GEE_PROJECT_ID)
    for key, area in STUDY_AREAS.items():
        explore(key, area)


if __name__ == "__main__":
    main()
