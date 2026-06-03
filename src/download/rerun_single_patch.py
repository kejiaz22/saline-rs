# -*- coding: utf-8 -*-
"""重跑单个 patch: patch_pingluo_002 (原始文件在 Drive 丢失)。

导出参数与 03_batch_export.py 完全一致 (8 波段 float32, NoData=-9999, 10m,
EPSG:4326), 坐标从 data/raw/batch_v1_metadata.csv 读取。

folder 用纯文件夹名 'saline_rerun_v1' (不带斜杠/子路径)。
提交成功后回写 metadata: 更新该行 task_id / status='RESUBMITTED' / notes。
提交后立即退出, 不轮询。

用法: conda run -n saline python src/download/rerun_single_patch.py
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import pandas as pd
import ee
from config import GEE_PROJECT_ID, DATE_START, DATE_END, DATA_RAW

# --- 与 03_batch_export.py 一致的参数 ---
HALF_DEG = 0.018
MAX_CLOUD = 20
NODATA = -9999
S2_BANDS = ["B2", "B3", "B4", "B8", "B11", "B12"]
S1_BANDS = ["VV", "VH"]

TARGET_FILE = "patch_pingluo_002.tif"
EXPORT_FOLDER = "saline_rerun_v1"          # 纯文件夹名, 不带斜杠
FILE_PREFIX = "patch_pingluo_002"
DESCRIPTION = "patch_pingluo_002_rerun"
RERUN_NOTE = "2026-06-02 rerun due to original missing from Drive"

META_CSV = DATA_RAW / "batch_v1_metadata.csv"
TASKS_URL = "https://code.earthengine.google.com/tasks"


def mask_s2_clouds(img: ee.Image) -> ee.Image:
    """SCL 去云: 3=云阴影, 8/9=云, 10=卷云。"""
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


def update_metadata(new_task_id: str) -> None:
    """回写 metadata: 更新目标行的 task_id / status / notes。"""
    meta = pd.read_csv(META_CSV)
    if "notes" not in meta.columns:
        meta["notes"] = ""
    meta["notes"] = meta["notes"].fillna("")

    mask = meta["filename"] == TARGET_FILE
    if not mask.any():
        raise RuntimeError(f"metadata 中找不到 {TARGET_FILE}")
    meta.loc[mask, "task_id"] = new_task_id
    meta.loc[mask, "status"] = "RESUBMITTED"
    meta.loc[mask, "notes"] = RERUN_NOTE

    meta.to_csv(META_CSV, index=False, encoding="utf-8-sig")


def main() -> None:
    ee.Initialize(project=GEE_PROJECT_ID)

    # 从 metadata 读坐标
    meta = pd.read_csv(META_CSV)
    row = meta.loc[meta["filename"] == TARGET_FILE]
    if row.empty:
        raise RuntimeError(f"metadata 中找不到 {TARGET_FILE}")
    lat = float(row["lat"].iloc[0])
    lon = float(row["lon"].iloc[0])
    print(f"读取坐标: lat={lat}, lon={lon}")

    bbox = build_bbox(lat, lon)
    stacked = build_s2(bbox).addBands(build_s1(bbox)).unmask(NODATA)  # 8 波段

    task = ee.batch.Export.image.toDrive(
        image=stacked,
        description=DESCRIPTION,
        folder=EXPORT_FOLDER,
        fileNamePrefix=FILE_PREFIX,
        scale=10,
        region=bbox,
        crs="EPSG:4326",
        maxPixels=int(1e9),
        formatOptions={"noData": NODATA},
    )
    task.start()

    update_metadata(task.id)

    print("重跑任务已提交 ✓")
    print(f"  新 task ID  : {task.id}")
    print(f"  Drive 文件夹 : {EXPORT_FOLDER}  (-> {FILE_PREFIX}.tif)")
    print(f"  metadata 已更新: status=RESUBMITTED, notes='{RERUN_NOTE}'")
    print(f"  监控页面    : {TASKS_URL}")
    print("提交后立即退出, 单 patch 通常 1-3 分钟完成。")


if __name__ == "__main__":
    main()
