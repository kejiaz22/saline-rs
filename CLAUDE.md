# 盐碱地遥感分类与识别项目

## 项目背景
- 研究者: Kejia (UC Davis, Environmental Science Data track)
- 时长: 原本 8 周课题,因时间紧压缩到 **2 周抢救版**
- 目标: 基于 Sentinel-1 SAR + Sentinel-2 光学数据,实现中国典型区域盐碱地分类,精度较传统方法 (SVM/RF) 提升 10% 以上

## 2 周抢救版的关键决策 (砍掉范围)
- 研究区: 原 4 个 → **缩减为 2 个** (黄河中上游沿黄灌区 + 东北松嫩-三江平原)
- 采样区: 原 1900 个 → **缩减为每区 30-50 个 patch** (共 60-100 个)
- 时间: 原"双期/年" → **单期** (2023 年植被生长季 6-9 月)
- 深度学习模型: 原 4 个 (RSMamba, DGCNet, EfficientViT-M2, ResNet50) → **缩减为 2 个 (ResNet50 + EfficientViT-M2)**
- 任务粒度: **patch 级二分类** (盐碱/非盐碱),不做像素级语义分割
- SAR + 光学融合: 简单波段堆叠,不做复杂融合

## 数据源
- Sentinel-2 (光学, 10m, 大气校正后 L2A) — 主力
- Sentinel-1 (SAR, 10m, GRD) — 辅助波段
- 真值参考: 中国土壤盐渍化 1:3200 万在线地图 (https://www.osgeo.cn/map/m01d8)

## 技术栈
- 数据获取: Google Earth Engine (Python API)
- 预处理: GEE 默认产品 + Python (rasterio, numpy)
- 标注: QGIS (patch 级,二分类)
- 传统分类: scikit-learn (SVM, Random Forest)
- 深度学习: PyTorch + torchvision + timm
- 实验跟踪: Weights & Biases (可选)
- 可视化: matplotlib, seaborn, QGIS

## 编码规范
- Python 3.10, conda 环境名: `saline`
- 路径用 `pathlib.Path`,根目录在 `config.py` 定义
- 中文注释 OK,函数名/变量名用英文
- 所有训练随机种子固定为 42
- 大文件不 commit (已在 .gitignore)

## 2 周时间线
- **Week A** (Day 1-7): 环境 + GEE 下载 + 预处理 + 标注 + 传统分类基准
- **Week B** (Day 8-14): 深度学习训练 + 评估 + 写报告

## 交付物
1. 一份完整研究报告 (IMRAD 结构)
2. 完整代码 (可复现)
3. 实际跑出的分类结果 (混淆矩阵、F1、Kappa)
4. 8 份周报 (补写,坦诚说明集中在最后两周完成)

## 当前状态
- ✅ 环境装好 (Anaconda, Git, VS Code, Claude Code, QGIS 3.34.11)
- ✅ Google Earth Engine 账号已有
- 🟡 进行中: 项目骨架搭建
- 🔲 下一步: 建 conda 环境 `saline`,装 Python 包,GEE 认证,写下载脚本
