# -*- coding: utf-8 -*-
"""调研: GEE 单文件导出是否支持 mixed dtype (S2 uint16 + S1 float32)。

思路: 构造一个极小的 8 波段 mixed-dtype 影像, 分两步取证:
  1. image.bandTypes() -- 看 GEE 内部是否保留每波段各自类型 (导出前)
  2. getDownloadURL 同步下载一个极小 GeoTIFF, 用 rasterio 看导出后真实 dtype

不走 Drive, 几秒出结果。

用法: conda run -n saline python src/utils/test_dtype.py
"""
import sys
import io
import zipfile
import urllib.request
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import rasterio
import ee
from config import GEE_PROJECT_ID, DATE_START, DATE_END

S2_BANDS = ["B2", "B3", "B4", "B8", "B11", "B12"]
S1_BANDS = ["VV", "VH"]


def build_mixed(bbox: ee.Geometry) -> ee.Image:
    """S2 -> uint16, S1 -> float32, 然后 stack。"""
    s2 = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterDate(DATE_START, DATE_END)
        .filterBounds(bbox)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
        .select(S2_BANDS)
        .median()
        .toUint16()  # S2 -> uint16
    )
    s1 = (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterDate(DATE_START, DATE_END)
        .filterBounds(bbox)
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
        .select(S1_BANDS)
        .mean()
        .toFloat()  # S1 -> float32
    )
    return s2.addBands(s1).clip(bbox)


def main() -> None:
    ee.Initialize(project=GEE_PROJECT_ID)

    # 极小方块 (~0.01 度), 让同步下载很快
    bbox = ee.Geometry.Rectangle([106.40, 38.89, 106.41, 38.90])
    img = build_mixed(bbox)

    # --- 步骤 1: GEE 内部各波段类型 (导出前) ---
    print("=== 步骤 1: image.bandTypes() (GEE 内部, 导出前) ===")
    bt = img.bandTypes().getInfo()
    for b in S2_BANDS + S1_BANDS:
        print(f"  {b:5s}: {bt[b]}")
    print()

    # --- 步骤 2: 同步下载 GeoTIFF, 看导出后真实 dtype ---
    print("=== 步骤 2: 实际导出 GeoTIFF 的 dtype (rasterio 读取) ===")
    url = img.getDownloadURL(
        {"scale": 10, "region": bbox, "format": "GEO_TIFF"}
    )
    print(f"  下载 URL 已生成, 拉取中...")
    raw = urllib.request.urlopen(url).read()

    # 可能是单 tif, 也可能是 zip; 两种都处理
    data = raw
    if raw[:2] == b"PK":  # zip 魔数
        zf = zipfile.ZipFile(io.BytesIO(raw))
        tif_name = [n for n in zf.namelist() if n.lower().endswith(".tif")][0]
        data = zf.read(tif_name)
        print(f"  (返回的是 zip, 内含 {tif_name})")

    with rasterio.MemoryFile(data) as mem, mem.open() as src:
        print(f"  波段数 : {src.count}")
        print(f"  逐波段 dtype: {src.dtypes}")
        unified = len(set(src.dtypes)) == 1
        print()
        print("=== 结论 ===")
        if unified:
            print(f"  GEE 把全部 8 波段统一成了单一 dtype: {src.dtypes[0]}")
            print("  -> 单文件 mixed dtype 不被支持 (GeoTIFF 格式限制)。")
        else:
            print(f"  各波段保留了不同 dtype -> 支持 mixed dtype!")


if __name__ == "__main__":
    main()
