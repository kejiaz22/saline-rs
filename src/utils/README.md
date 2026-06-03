# src/utils — 一次性诊断 / 调研工具

放置排查问题、验证假设用的临时脚本。区别于 `src/download/`(真正的下载逻辑)。
这些脚本不在主流程里调用,需要时手动跑。

## 脚本说明

### `diagnose_patch.py`
检查某个导出的 patch tif 是否正常(排查"QGIS 显示空白"一类问题)。
打印文件大小、尺寸、波段数、dtype、CRS、地理范围,以及逐波段的
min/max/mean 和空值占比,并自动判断空白原因(导出失败 / 拉伸问题 / 波段堆叠出错)。

```bash
conda run -n saline python src/utils/diagnose_patch.py <文件名>
# 例: conda run -n saline python src/utils/diagnose_patch.py test_patch_pingluo_001.tif
```

### `test_dtype.py`
调研 GEE 单文件导出是否支持 mixed dtype。
结论:**不支持** —— GeoTIFF 格式要求同文件所有波段共享一个 dtype,
GEE 会把 S2(uint16)+ S1(float32) 统一向上转换成 float32。
因此项目最终采用「8 波段全 float32 单文件」方案。

```bash
conda run -n saline python src/utils/test_dtype.py
```
