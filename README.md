# 盐碱地遥感分类与识别

基于 Sentinel-1 SAR + Sentinel-2 光学数据,对中国典型区域(沿黄灌区、松嫩-三江平原)进行 patch 级盐碱地二分类,对比传统方法(SVM/RF)与深度学习模型(ResNet50 / EfficientViT-M2)的精度。

> 本仓库为 8 周课题的 **2 周抢救版**。范围决策与时间线详见 [CLAUDE.md](CLAUDE.md)。

## 目录结构

```
saline-rs/
├── data/
│   ├── raw/          # GEE 下载的原始影像 (不 commit)
│   ├── processed/    # 预处理后的 patch (不 commit)
│   └── labels/       # patch 级标注 (二分类)
├── notebooks/        # 探索与可视化
├── src/
│   ├── download/     # GEE 下载脚本
│   ├── preprocess/   # 影像裁剪、波段堆叠、patch 切分
│   ├── classical/    # SVM / Random Forest 基准
│   ├── deeplearning/ # ResNet50 / EfficientViT-M2 训练
│   └── eval/         # 混淆矩阵、F1、Kappa 评估
├── reports/          # 研究报告与周报
└── results/          # 模型权重、图表、输出
```

## 快速开始

```bash
conda create -n saline python=3.10
conda activate saline
# 后续: 安装依赖、GEE 认证、运行下载脚本
```
