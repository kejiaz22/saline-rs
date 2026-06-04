---
## 项目当前状态 (Day 2 结束, 2026-06-03)

✅ Week 1-4 全部完成
🔲 Week 5 待开始 (深度学习实现)

### 进度
- 总进度: ~50%
- 剩余: 12-13 天
- 下一步: ResNet50 / EfficientViT-M2 + 迁移学习 + 5-fold CV

### 待决策 (Day 3 优先)
- 深度学习输入通道方案: 8 波段 tif vs 3 通道 PNG
- 训练硬件: 本地 CPU vs Google Colab Pro GPU
- 数据增强策略具体参数

### 基线性能 (传统方法)
- SVM F1=0.544 ± 0.219
- RF F1=0.564 ± 0.217
- 深度学习目标 F1 ≥ 0.65
---

# Week 02 周报 (Day 2)

日期: 2026-06-03 ~ 2026-06-04

## 完成
1. 指数策略 v2: 植被惩罚 (×0.4) + 重新权重 (low_ndvi 升至 0.35), salinity_prior std 0.10→0.19
2. 自研网页标注工具 (Flask + 原生 JS + Tailwind, 键盘快捷键加速)
3. 60 patch 标注完成 (21 saline / 39 non-saline, 0 skip, 类别比 ~1:1.9 健康)
4. 特征工程: 34 个特征 (24 光谱统计 mean/std/p25/p75 + 4 SAR + 6 衍生指数)
5. SVM (RBF) baseline: 5×10 RepeatedStratifiedKFold + nested GridSearch
6. RF baseline: 5×10 RepeatedSKF + feature importance + 与 SVM 对比

## 关键发现
1. NDSI 公式 (B4-B8)/(B4+B8) 与 NDVI 数学冗余 (恒为 -NDVI), Day 1 发现, Day 2 改用 SWIR 盐分指数
2. SWIR 盐分先验方向相反: SI / NDSI-SWIR / SR-SWIR 与人工标签呈负相关 (rho ~ -0.30),
   "SWIR 高 = 盐碱" 的文献先验在本地真值中不成立 → Discussion 重点
3. 消融实验反直觉: 剔除 16 个 low-confidence 样本反而使 F1 下降 ~0.14 (SVM 与 RF 一致),
   小样本下样本数量比标签纯度更重要
4. SVM 与 RF 在 95% CI 内统计无显著差异 (打平), 平均 F1(saline) ≈ 0.55
5. RF feature importance Top 5 全是光谱统计量 (B8_p25, B2_std, B4_p25, B11_p25, B3_p25),
   衍生指数排名靠后 (ndsi_swir #8); 重要性高度摊平, 无主导特征

## 量化结果 (全样本, 5×10 RepeatedSKF)
| 指标 | SVM [95% CI] | RF [95% CI] |
|------|-------------|-------------|
| Accuracy | 0.712 [0.676, 0.747] | 0.753 [0.722, 0.784] |
| F1 (saline) | 0.544 [0.483, 0.606] | 0.564 [0.504, 0.624] |
| F1 macro | 0.663 | 0.694 |

## 工程产出 (累计)
- 12 次 git commit (Day 1-2, 含本周报)
- 16 个 Python 脚本 + 1 个 __init__.py (下载/预处理/标注/分类/工具)
- 完整 pipeline: 60 patch 数据获取 → 质量验证 → 标注 → 特征 → 传统分类基线

## 待办
- Day 3-4: 深度学习 (ResNet50 + EfficientViT-M2), ImageNet 预训练 + 微调 + 强增强 + 5-fold CV
- 目标: 在 F1 ≈ 0.55 基线上提升 ≥10%, 即 F1 ≥ 0.65

## 已知问题
- RF 训练在本地 conda 环境有 OpenMP 段错误 (0xC0000005), 非确定性;
  决策: 接受"重试到成功", random_state=42 保证成功run的数值可复现 (不硬化环境)
- RF 因崩溃改用"一次性选超参 + 固定超参 repeated CV" (SVM 仍 nested), 方法学略不对称, 已记录
- 单期生长季光学方案对植被覆盖下的盐碱信号有固有识别局限
