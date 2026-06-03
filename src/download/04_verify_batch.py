# -*- coding: utf-8 -*-
"""批量验证: 核对 batch_v1 下载的 60 个 patch 是否齐全 + 逐个体检。

以 data/raw/batch_v1_metadata.csv 为准 (期望文件清单), 对每个文件检查:
  - 是否存在 (缺失 -> MISSING)
  - 能否用 rasterio 打开 (打不开 -> FAIL)
  - 波段数是否为 8, dtype 是否全 float32 (不符 -> FAIL)
  - NoData 是否为 -9999 (不符 -> WARN)
  - 尺寸是否约 400x400 (偏差大 -> WARN)
  - NoData 像元占比 (过高 -> WARN, 可能是空/水体 patch)

最后打印汇总: OK / WARN / FAIL / MISSING 各几个, 并列出非 OK 的文件,
同时把逐 patch 明细写入 data/raw/batch_v1_quality_report.csv。

用法: conda run -n saline python src/download/04_verify_batch.py
"""
import sys
from pathlib import Path

# 把项目根目录加入 sys.path 以便 import config
sys.path.append(str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import rasterio

from config import DATA_RAW

# --- 期望规格 ---
BATCH_DIR = DATA_RAW / "saline_export_batch_v1"
META_CSV = DATA_RAW / "batch_v1_metadata.csv"
REPORT_CSV = DATA_RAW / "batch_v1_quality_report.csv"
EXPECT_BANDS = 8
EXPECT_DTYPE = "float32"
EXPECT_NODATA = -9999
EXPECT_SIZE = 400          # 约 400x400 (4km @ 10m)
SIZE_TOL = 20              # 尺寸容差 (像元)
NODATA_WARN_PCT = 50.0     # NoData 占比超过此值则 WARN


def check_one(tif_path: Path) -> dict:
    """返回逐 patch 指标 dict, 含 verdict ∈ {OK, WARN, FAIL, MISSING}。"""
    rec = {
        "filename": tif_path.name,
        "verdict": "",
        "bands": None,
        "dtype": "",
        "nodata": None,
        "width": None,
        "height": None,
        "nodata_pct": None,
        "detail": "",
    }

    if not tif_path.exists():
        rec["verdict"] = "MISSING"
        rec["detail"] = "文件不存在"
        return rec

    try:
        with rasterio.open(tif_path) as src:
            problems = []   # FAIL 级
            warnings = []   # WARN 级

            rec["bands"] = src.count
            rec["dtype"] = src.dtypes[0] if len(set(src.dtypes)) == 1 else str(src.dtypes)
            rec["nodata"] = src.nodata
            rec["width"] = src.width
            rec["height"] = src.height

            if src.count != EXPECT_BANDS:
                problems.append(f"波段数={src.count} (应为 {EXPECT_BANDS})")
            if any(dt != EXPECT_DTYPE for dt in src.dtypes):
                problems.append(f"dtype={src.dtypes} (应全为 {EXPECT_DTYPE})")

            nd = src.nodata
            if nd is None or int(nd) != EXPECT_NODATA:
                warnings.append(f"NoData={nd} (应为 {EXPECT_NODATA})")

            if (
                abs(src.width - EXPECT_SIZE) > SIZE_TOL
                or abs(src.height - EXPECT_SIZE) > SIZE_TOL
            ):
                warnings.append(f"尺寸={src.width}x{src.height}")

            # NoData 占比 (以第 1 波段为代表)
            band1 = src.read(1)
            ref_nd = nd if nd is not None else EXPECT_NODATA
            nd_frac = 100.0 * np.sum(band1 == ref_nd) / band1.size
            rec["nodata_pct"] = round(nd_frac, 2)
            if nd_frac > NODATA_WARN_PCT:
                warnings.append(f"NoData占比={nd_frac:.1f}%")

            if problems:
                rec["verdict"] = "FAIL"
                rec["detail"] = "; ".join(problems + warnings)
            elif warnings:
                rec["verdict"] = "WARN"
                rec["detail"] = "; ".join(warnings)
            else:
                rec["verdict"] = "OK"
                rec["detail"] = f"8 波段 float32, NoData占比 {nd_frac:.1f}%"
    except Exception as e:  # noqa: BLE001 — 打不开就算 FAIL
        rec["verdict"] = "FAIL"
        rec["detail"] = f"打开失败: {e}"
    return rec


def main() -> None:
    if not META_CSV.exists():
        print(f"!! 找不到 metadata: {META_CSV}")
        return

    meta = pd.read_csv(META_CSV)
    print(f"metadata 期望文件数: {len(meta)}")
    print(f"验证目录: {BATCH_DIR}")
    print("=" * 64)

    records = [check_one(BATCH_DIR / fname) for fname in meta["filename"]]

    # 写逐 patch 质量报告 CSV
    report = pd.DataFrame(records)
    report.to_csv(REPORT_CSV, index=False, encoding="utf-8-sig")

    # 逐个列出非 OK 的
    counts = {"OK": 0, "WARN": 0, "FAIL": 0, "MISSING": 0}
    for rec in records:
        counts[rec["verdict"]] += 1
    for level in ("MISSING", "FAIL", "WARN"):
        for rec in records:
            if rec["verdict"] == level:
                print(f"[{level:7s}] {rec['filename']}  --  {rec['detail']}")

    print("=" * 64)
    # NoData 占比统计 (仅对能打开的文件)
    nd_vals = [r["nodata_pct"] for r in records if r["nodata_pct"] is not None]
    if nd_vals:
        lt5 = sum(1 for v in nd_vals if v < 5.0)
        print(
            f"NoData 占比: <5% 的有 {lt5}/{len(nd_vals)} 个, "
            f"最大 {max(nd_vals):.1f}%, 中位 {float(np.median(nd_vals)):.1f}%"
        )
    print(
        f"汇总: OK={counts['OK']}  WARN={counts['WARN']}  "
        f"FAIL={counts['FAIL']}  MISSING={counts['MISSING']}  (共 {len(meta)})"
    )
    print(f"可用 (OK+WARN): {counts['OK'] + counts['WARN']}/{len(meta)}")
    print(f"逐 patch 报告: {REPORT_CSV}")
    if counts["MISSING"] or counts["FAIL"]:
        print("-> 存在缺失/损坏文件, 建议从 Drive 重新下载或重跑对应任务。")
    else:
        print("-> 全部文件可用。")


if __name__ == "__main__":
    main()
