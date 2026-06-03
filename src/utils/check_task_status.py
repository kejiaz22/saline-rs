# -*- coding: utf-8 -*-
"""诊断: 查 batch_v1 全部任务在 GEE 端的真实状态。

读 data/raw/batch_v1_metadata.csv 里的 task_id, 调 ee.data.getTaskStatus 批量查询,
先单独打印目标任务 (Task A), 再列出所有非 COMPLETED 的任务 (Task B)。

用法: conda run -n saline python src/utils/check_task_status.py
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import pandas as pd
import ee
from config import GEE_PROJECT_ID, DATA_RAW

META_CSV = DATA_RAW / "batch_v1_metadata.csv"
FOCUS_TASK = "YAWL3IMPEQRXTI2H6A7F6IUL"  # patch_pingluo_002


def main() -> None:
    ee.Initialize(project=GEE_PROJECT_ID)
    meta = pd.read_csv(META_CSV)
    id2name = dict(zip(meta["task_id"], meta["filename"]))
    task_ids = list(meta["task_id"])

    # 批量查询 (分块以防一次太多)
    statuses = []
    for i in range(0, len(task_ids), 25):
        statuses.extend(ee.data.getTaskStatus(task_ids[i : i + 25]))

    by_id = {s["id"]: s for s in statuses}

    # --- Task A: 目标任务 ---
    print("=== Task A: patch_pingluo_002 ===")
    s = by_id.get(FOCUS_TASK, {"id": FOCUS_TASK, "state": "NOT_FOUND"})
    print(s)
    print()

    # --- Task B: 全部状态统计 + 非 COMPLETED 列表 ---
    print("=== Task B: 全部 60 个任务 ===")
    counts = {}
    bad = []
    for tid in task_ids:
        st = by_id.get(tid, {"state": "NOT_FOUND"})
        state = st.get("state", "UNKNOWN")
        counts[state] = counts.get(state, 0) + 1
        if state != "COMPLETED":
            bad.append((id2name.get(tid, "?"), tid, state, st.get("error_message", "")))

    print("状态统计:", counts)
    print()
    if bad:
        print(f"非 COMPLETED 任务 ({len(bad)} 个):")
        for fname, tid, state, err in bad:
            line = f"  [{state}] {fname}  ({tid})"
            if err:
                line += f"  -- {err}"
            print(line)
    else:
        print("全部 60 个任务均为 COMPLETED。")


if __name__ == "__main__":
    main()
