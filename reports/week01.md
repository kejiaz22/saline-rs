# Week 01 周报 (Day 1)

日期: 2026-06-02

## 完成
- 环境搭建: conda 环境 `saline`、关键包验证、Google Earth Engine 认证通过
- 数据获取: 两个研究区 (宁夏平罗西大滩 / 吉林大安东) 各 30 个 4km×4km patch,
  共 60 个,8 波段 (S2 6 + S1 2),float32 + NoData=-9999
- 质量验证: 60/60 文件通过完整性检查 (波段/类型/尺寸/无空洞)
- 预处理: 计算 6 个光谱/SAR 指数 (NDVI, NDWI, SI, NDSI-SWIR, SR-SWIR, VH/VV)

## 发现 / 踩坑
- 原 NDSI 公式 (B4-B8)/(B4+B8) 恒等于 -NDVI,信息冗余,已弃用,改用 SWIR 盐分指数
- SWIR 启发式当前过滤未生效 (SR-SWIR>1.1 恒真; SI>median 未起作用),候选退化为裸土主导
- GEE 的 folder 参数不支持子目录路径 (斜杠),需用纯文件夹名
- patch_pingluo_002 原始导出在 Drive 丢失 → 用 rerun_single_patch.py 单独重跑修复,
  metadata 回写 RESUBMITTED + notes

## 明日
- 决定指数策略 (方案 A 快修 / B 连续打分 / C 引入真值地图)
- QGIS patch 级目视标注 (二分类)
- 传统分类基准 (SVM / Random Forest)
