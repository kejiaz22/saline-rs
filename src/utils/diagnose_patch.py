# -*- coding: utf-8 -*-
"""临时诊断脚本: 检查导出的 patch tif 是否正常 (空白问题排查)。

用法:
  conda run -n saline python src/utils/diagnose_patch.py <文件名>
  默认检查 test_patch_pingluo_001.tif

打印基本信息 + 逐波段统计 + 自动判断空白原因。
"""
import sys
from pathlib import Path

# 把项目根目录加入 sys.path 以便 import config
sys.path.append(str(Path(__file__).resolve().parents[2]))

import numpy as np
import rasterio

from config import DATA_RAW


def diagnose(tif_path: Path) -> None:
    print("=" * 60)
    print(f"诊断文件: {tif_path.name}")
    print("=" * 60)

    if not tif_path.exists():
        print(f"!! 文件不存在: {tif_path}")
        return

    size_mb = tif_path.stat().st_size / (1024 * 1024)
    print(f"文件大小 : {size_mb:.2f} MB")

    with rasterio.open(tif_path) as src:
        print(f"尺寸     : {src.width} x {src.height} (width x height)")
        print(f"波段数   : {src.count}")
        print(f"数据类型 : {src.dtypes[0]}")
        print(f"CRS      : {src.crs}")
        print(f"地理范围 : {src.bounds}")
        print(f"NoData   : {src.nodata}")
        print("-" * 60)

        band_flags = []  # 每个波段是否"全空" (全 0 或全 NaN)
        for i in range(1, src.count + 1):
            band = src.read(i).astype("float64")
            total = band.size

            nan_cnt = int(np.isnan(band).sum())
            zero_cnt = int((band == 0).sum())
            empty_cnt = nan_cnt + zero_cnt  # 视 0 与 NaN 同为"空"
            empty_pct = 100.0 * empty_cnt / total

            valid = band[~np.isnan(band) & (band != 0)]
            if valid.size > 0:
                vmin, vmax, vmean = valid.min(), valid.max(), valid.mean()
            else:
                vmin = vmax = vmean = float("nan")

            is_empty = valid.size == 0
            band_flags.append(is_empty)

            print(
                f"波段 {i}: min={vmin:.3f}  max={vmax:.3f}  mean={vmean:.3f}  "
                f"| NaN={100.0*nan_cnt/total:.1f}%  0值={100.0*zero_cnt/total:.1f}%  "
                f"空值合计={empty_pct:.1f}%"
            )

        print("-" * 60)
        # 自动判断
        n_empty = sum(band_flags)
        n_total = len(band_flags)
        if n_empty == n_total:
            print("判断: 所有波段都是 0/NaN -> 导出失败, 可能 region 错了或合成为空。")
        elif n_empty == 0:
            print("判断: 所有波段都有有效值 -> 文件本身正常, 大概率是 QGIS 拉伸显示问题。")
        else:
            print(
                f"判断: {n_empty}/{n_total} 个波段全空, 其余有值 "
                f"-> 波段堆叠出错 (部分数据源没叠进来)。"
            )
    print()


def main() -> None:
    fname = sys.argv[1] if len(sys.argv) > 1 else "test_patch_pingluo_001.tif"
    diagnose(DATA_RAW / fname)


if __name__ == "__main__":
    main()
