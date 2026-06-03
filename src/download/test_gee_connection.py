# -*- coding: utf-8 -*-
"""快速验证 GEE 连通性: 每次新 session 跑一次即可。

用法: conda run -n saline python src/download/test_gee_connection.py
"""
import sys
from pathlib import Path

# 把项目根目录加入 sys.path 以便 import config
sys.path.append(str(Path(__file__).resolve().parents[2]))

import ee
from config import GEE_PROJECT_ID

ee.Initialize(project=GEE_PROJECT_ID)
print("GEE connection works ✓")
