# -*- coding: utf-8 -*-
"""项目全局配置: 路径与常量。"""
from pathlib import Path

# 关键路径
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
DATA_LABELS = PROJECT_ROOT / "data" / "labels"
RESULTS_DIR = PROJECT_ROOT / "results"
REPORTS_DIR = PROJECT_ROOT / "reports"
CHECKPOINTS_DIR = RESULTS_DIR / "checkpoints"

# 全局常量
RANDOM_SEED = 42

# Google Earth Engine 项目 ID
GEE_PROJECT_ID = "direct-archery-473217-s8"

# 两个研究区中心坐标 (粗略, 后续精化)
STUDY_AREAS = {
    "huanghe": {"name": "黄河中上游沿黄灌区 (银川附近)", "lat": 38.5, "lon": 106.2},
    "songnen": {"name": "东北松嫩-三江平原 (大安附近)", "lat": 45.5, "lon": 124.0},
}

# 时间范围: 2023 年植被生长季
DATE_START = "2023-06-01"
DATE_END = "2023-09-30"
