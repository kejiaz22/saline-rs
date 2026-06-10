# -*- coding: utf-8 -*-
"""深度学习训练超参数 (与项目根 config.py 区分)。

通过 importlib 按路径载入项目根 config.py (避免同名 import 冲突),
继承其中的路径与随机种子, 再补充训练专用超参。
"""
import importlib.util
from pathlib import Path

import torch

# --- 载入项目根 config.py (按文件路径, 命名为 project_config 避免冲突) ---
_PROJ_CONFIG = Path(__file__).resolve().parents[2] / "config.py"
_spec = importlib.util.spec_from_file_location("project_config", _PROJ_CONFIG)
_pc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_pc)

# 路径 (继承自项目 config)
DATA_RAW = _pc.DATA_RAW
DATA_LABELS = _pc.DATA_LABELS
DATA_PROCESSED = _pc.DATA_PROCESSED
RESULTS_DIR = _pc.RESULTS_DIR
CHECKPOINTS_DIR = _pc.CHECKPOINTS_DIR
RANDOM_SEED = _pc.RANDOM_SEED

# --- 输入规格 ---
IN_CHANNELS = 8                 # 8 波段堆叠
INPUT_SIZE = 224                # ResNet 标准输入
NUM_CLASSES = 2                 # 盐碱 / 非盐碱
NODATA = -9999
# 8 波段顺序 (与 tif 堆叠 + features.csv 一致)
BAND_NAMES = ["B2", "B3", "B4", "B8", "B11", "B12", "VV", "VH"]

# --- 训练超参 ---
BATCH_SIZE = 8
EPOCHS = 50
# v1 不稳定调参 (基于 5-fold 发现: F1 std>mean, Fold5 best_epoch=0 即初始化即最优):
LR = 5e-5                       # 1e-4 -> 5e-5: 降学习率, 减少"初始化即最优"早停现象
WEIGHT_DECAY = 1e-3
T_MAX = 50                      # CosineAnnealingLR
EARLY_STOP_PATIENCE = 20        # 10 -> 20: 给模型更多机会学 (避免过早早停)
N_SPLITS = 5
N_REPEATS = 3                   # 5x3=15 fold 估计, 收窄 CI (Change 1)

# --- 结果文件 ---
RESULTS_JSON = RESULTS_DIR / "resnet50_results.json"
SVM_JSON = RESULTS_DIR / "svm_results_repeated.json"
RF_JSON = RESULTS_DIR / "rf_results.json"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
