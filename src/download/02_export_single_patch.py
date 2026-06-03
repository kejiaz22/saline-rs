# -*- coding: utf-8 -*-
"""测试性导出: 导出两个研究区各 1 个 patch, 验证整条导出流程 (不批量)。

两个 4km×4km 测试方块 (中心坐标读自 config.STUDY_AREAS):
  - 黄河中上游 (宁夏平罗西大滩) -> test_patch_pingluo_001
  - 东北松嫩   (吉林大安东)     -> test_patch_daan_001

每个 patch 制作 S2(6 波段)+S1(2 波段) 共 8 波段合成影像, 提交到 Google Drive。
提交后立即退出, 不轮询等待。

波段顺序: B2 B3 B4 B8 B11 B12 (S2) + VV VH (S1)

用法: conda run -n saline python src/download/02_export_single_patch.py
"""
import sys
from pathlib import Path

# 把项目根目录加入 sys.path 以便 import config
sys.path.append(str(Path(__file__).resolve().parents[2]))

import ee
from config import GEE_PROJECT_ID, STUDY_AREAS, DATE_START, DATE_END

HALF_DEG = 0.018  # 约 4km/2, 总边长 ~0.036 度 ≈ 4km
MAX_CLOUD = 20

S2_BANDS = ["B2", "B3", "B4", "B8", "B11", "B12"]
S1_BANDS = ["VV", "VH"]

EXPORT_FOLDER = "saline_export"
TASKS_URL = "https://code.earthengine.google.com/tasks"

# 研究区 key -> 导出文件名前缀
PATCH_NAMES = {
    "huanghe": "test_patch_pingluo_001",
    "songnen": "test_patch_daan_001",
}


def mask_s2_clouds(img: ee.Image) -> ee.Image:
    """用 SCL 波段去除云、云阴影、卷云。

    SCL 类别: 3=云阴影, 8=云(中概率), 9=云(高概率), 10=卷云。
    """
    scl = img.select("SCL")
    mask = scl.neq(3).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10))
    return img.updateMask(mask)


def build_s2(bbox: ee.Geometry) -> ee.Image:
    """S2 SR: 去云 -> 取 6 波段 -> 中位数合成 -> clip。"""
    col = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterDate(DATE_START, DATE_END)
        .filterBounds(bbox)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", MAX_CLOUD))
        .map(mask_s2_clouds)
    )
    return col.select(S2_BANDS).median().clip(bbox)


def build_s1(bbox: ee.Geometry) -> ee.Image:
    """S1 GRD: IW + VV/VH -> 取 2 波段 -> 均值合成 -> clip。"""
    col = (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterDate(DATE_START, DATE_END)
        .filterBounds(bbox)
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
    )
    return col.select(S1_BANDS).mean().clip(bbox)


def build_bbox(lat: float, lon: float) -> ee.Geometry:
    """以 (lon, lat) 为中心, 各扩 HALF_DEG 度构造矩形。"""
    return ee.Geometry.Rectangle(
        [lon - HALF_DEG, lat - HALF_DEG, lon + HALF_DEG, lat + HALF_DEG]
    )


def export_patch(key: str, area: dict) -> str:
    """构建 8 波段合成并提交导出, 返回 task ID。"""
    name = PATCH_NAMES[key]
    bbox = build_bbox(area["lat"], area["lon"])
    stacked = build_s2(bbox).addBands(build_s1(bbox))  # S2 6 + S1 2 = 8 波段

    task = ee.batch.Export.image.toDrive(
        image=stacked,
        description=name,
        folder=EXPORT_FOLDER,
        fileNamePrefix=name,
        scale=10,
        region=bbox,
        crs="EPSG:4326",
        maxPixels=int(1e9),
    )
    task.start()

    print(f"导出任务已提交 ✓  [{area['name']}]")
    print(f"  description : {name}")
    print(f"  task ID     : {task.id}")
    print(f"  目标位置    : Google Drive / {EXPORT_FOLDER}/{name}.tif")
    print()
    return task.id


def main() -> None:
    ee.Initialize(project=GEE_PROJECT_ID)

    task_ids = {}
    for key, area in STUDY_AREAS.items():
        task_ids[PATCH_NAMES[key]] = export_patch(key, area)

    print(f"波段 (8) : {S2_BANDS + S1_BANDS}")
    print(f"监控页面 : {TASKS_URL}")
    print("两个任务均已提交, GEE 后台运行, 请去 Drive 查看结果。")


if __name__ == "__main__":
    main()
