# -*- coding: utf-8 -*-
"""批量导出: 两个研究区各 30 个 patch, 共 60 个 (不轮询等待)。

采样策略 (可复现, 种子 = config.RANDOM_SEED):
  - 每区在中心点 ±0.25 度 (约 50km×50km) 内随机撒 30 个 4km×4km patch
  - patch 中心两两至少相距 MIN_DIST_KM = 6km (不重叠, 避免空间相关泄漏)
  - 用 ee.Geometry 校验 patch 完整落在中国大陆 (不出海)
  - patch 完整落在 ROI 范围内 (采样时即收紧中心可取范围)

导出参数与 v2 一致: 8 波段 float32, NoData=-9999, 10m, EPSG:4326。

注意: GEE 的 folder 参数不支持嵌套路径, 故用单层名 'saline_export_batch_v1'。

提交后写本地 metadata CSV (data/raw/batch_v1_metadata.csv) 记录每个 patch 的
文件名 / lat / lon / region / task_id / status, 便于事后追溯。

用法: conda run -n saline python src/download/03_batch_export.py
"""
import sys
import math
from pathlib import Path

# 把项目根目录加入 sys.path 以便 import config
sys.path.append(str(Path(__file__).resolve().parents[2]))

import random
import pandas as pd
import ee
from config import (
    GEE_PROJECT_ID,
    STUDY_AREAS,
    DATE_START,
    DATE_END,
    DATA_RAW,
    RANDOM_SEED,
)

# --- patch / 采样参数 ---
HALF_DEG = 0.018          # 单 patch 半边长, 总边长 ~0.036 度 ≈ 4km
ROI_HALF_DEG = 0.25       # 采样 ROI 半边长, 总边长 0.5 度 ≈ 50km
MAX_CLOUD = 20
NODATA = -9999
MIN_DIST_KM = 6.0         # patch 中心最小间距
N_PATCHES = 30            # 每区 patch 数
MAX_ATTEMPTS = 100000     # 采样上限 (防死循环)

S2_BANDS = ["B2", "B3", "B4", "B8", "B11", "B12"]
S1_BANDS = ["VV", "VH"]

EXPORT_FOLDER = "saline_export_batch_v1"
META_CSV = DATA_RAW / "batch_v1_metadata.csv"
TASKS_URL = "https://code.earthengine.google.com/tasks"

# 研究区 key -> 文件名前缀
REGION_PREFIX = {"huanghe": "pingluo", "songnen": "daan"}


# ============ 影像合成 (与 02_export_single_patch.py 一致) ============
def mask_s2_clouds(img: ee.Image) -> ee.Image:
    """用 SCL 波段去除云、云阴影、卷云 (3=云阴影, 8/9=云, 10=卷云)。"""
    scl = img.select("SCL")
    mask = scl.neq(3).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10))
    return img.updateMask(mask)


def build_s2(bbox: ee.Geometry) -> ee.Image:
    col = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterDate(DATE_START, DATE_END)
        .filterBounds(bbox)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", MAX_CLOUD))
        .map(mask_s2_clouds)
    )
    return col.select(S2_BANDS).median().clip(bbox).toFloat()


def build_s1(bbox: ee.Geometry) -> ee.Image:
    col = (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterDate(DATE_START, DATE_END)
        .filterBounds(bbox)
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
    )
    return col.select(S1_BANDS).mean().clip(bbox).toFloat()


def build_bbox(lat: float, lon: float) -> ee.Geometry:
    return ee.Geometry.Rectangle(
        [lon - HALF_DEG, lat - HALF_DEG, lon + HALF_DEG, lat + HALF_DEG]
    )


# ============ 采样 ============
def km_between(lat1, lon1, lat2, lon2) -> float:
    """等距圆柱近似下的地表距离 (km), 6km 量级足够精确。"""
    dlat = (lat2 - lat1) * 111.32
    dlon = (lon2 - lon1) * 111.32 * math.cos(math.radians((lat1 + lat2) / 2))
    return math.hypot(dlat, dlon)


def china_geometry() -> ee.Geometry:
    """中国大陆边界 (LSIB), 用于校验 patch 不出海。"""
    return (
        ee.FeatureCollection("USDOS/LSIB_SIMPLE/2017")
        .filter(ee.Filter.eq("country_na", "China"))
        .geometry()
    )


def batch_on_land(centers, china) -> list:
    """对一批中心点的 patch 做服务端 contains 校验, 单次 getInfo 返回布尔列表。"""
    feats = [ee.Feature(build_bbox(lat, lon)) for lat, lon in centers]
    fc = ee.FeatureCollection(feats)
    fc = fc.map(lambda ft: ft.set("onland", china.contains(ft.geometry(), ee.ErrorMargin(1))))
    return fc.aggregate_array("onland").getInfo()


def sample_centers(area: dict, china) -> list:
    """在 ROI 内随机撒 N_PATCHES 个满足间距 + 落陆 的中心点。"""
    # 收紧中心可取范围, 保证整 patch 落在 ROI 内
    span = ROI_HALF_DEG - HALF_DEG
    lat0, lon0 = area["lat"], area["lon"]
    lo_lat, hi_lat = lat0 - span, lat0 + span
    lo_lon, hi_lon = lon0 - span, lon0 + span

    accepted = []
    attempts = 0
    while len(accepted) < N_PATCHES:
        # 先用本地间距约束填满候选
        while len(accepted) < N_PATCHES and attempts < MAX_ATTEMPTS:
            attempts += 1
            lat = random.uniform(lo_lat, hi_lat)
            lon = random.uniform(lo_lon, hi_lon)
            if all(
                km_between(lat, lon, a_lat, a_lon) >= MIN_DIST_KM
                for a_lat, a_lon in accepted
            ):
                accepted.append((lat, lon))
        if attempts >= MAX_ATTEMPTS and len(accepted) < N_PATCHES:
            raise RuntimeError(
                f"采样失败: {MAX_ATTEMPTS} 次内只放下 {len(accepted)}/{N_PATCHES} 个 patch"
            )
        # 服务端落陆校验, 去掉出海的, 循环回去补
        flags = batch_on_land(accepted, china)
        accepted = [c for c, ok in zip(accepted, flags) if ok]

    return accepted


# ============ 导出 ============
def export_patch(name: str, lat: float, lon: float) -> str:
    bbox = build_bbox(lat, lon)
    stacked = build_s2(bbox).addBands(build_s1(bbox)).unmask(NODATA)  # 8 波段
    task = ee.batch.Export.image.toDrive(
        image=stacked,
        description=name,
        folder=EXPORT_FOLDER,
        fileNamePrefix=name,
        scale=10,
        region=bbox,
        crs="EPSG:4326",
        maxPixels=int(1e9),
        formatOptions={"noData": NODATA},
    )
    task.start()
    return task.id


def main() -> None:
    ee.Initialize(project=GEE_PROJECT_ID)
    random.seed(RANDOM_SEED)  # 固定种子, 全程可复现
    china = china_geometry()

    rows = []
    for key, area in STUDY_AREAS.items():
        prefix = REGION_PREFIX[key]
        print(f"[{area['name']}] 采样 {N_PATCHES} 个 patch ...")
        centers = sample_centers(area, china)

        for i, (lat, lon) in enumerate(centers, start=1):
            name = f"patch_{prefix}_{i:03d}"
            task_id = export_patch(name, lat, lon)
            rows.append(
                {
                    "filename": f"{name}.tif",
                    "lat": round(lat, 6),
                    "lon": round(lon, 6),
                    "region": prefix,
                    "task_id": task_id,
                    "status": "SUBMITTED",
                }
            )
        print(f"  -> 已提交 {len(centers)} 个任务")

    # 写 metadata CSV
    META_CSV.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(META_CSV, index=False, encoding="utf-8-sig")

    print()
    print(f"共提交任务   : {len(rows)} 个 (每区 {N_PATCHES})")
    print(f"metadata CSV : {META_CSV}")
    print(f"Drive 文件夹  : {EXPORT_FOLDER} (GEE 不支持嵌套路径, 用单层名分组)")
    print(f"监控页面     : {TASKS_URL}")
    print("预计 30-60 分钟全部完成。提交后立即退出, 不轮询。")


if __name__ == "__main__":
    main()
