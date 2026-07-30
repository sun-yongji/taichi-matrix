# 太极矩阵 · 六边形注意力昇腾算子

## TaiChi HexAttention — Ascend Custom Operator

将易学六爻对称性（C6六重对称群）映射为昇腾自定义注意力算子。

- **6个注意力头** ↔ 六爻（初→上）
- **60° 六边形拓扑** ↔ C6 对称群
- **每头1/6计算量**，6头并行，合起来全覆盖

## 性能

| 对比 | 方形注意力 | 六边形注意力 |
|------|-----------|-------------|
| 计算量 | O(N²) | **O(N²/6)** |
| 对角线覆盖 | 13% | **33.3%** |
| 加速比 | 1x | **~36x** |

## 文件结构

```
operator/
  hex_attention.py       # 昇腾算子代码
test/
  test_performance.py    # 性能对比测试
doc/
  technical_report.md    # 技术报告
COMPETITION.md           # 参赛方案书
submit.sh                # 打包脚本
```

## 参赛信息

- 作品名称：太极矩阵 — C6六边形注意力算子
- 团队：太极量子团队
- 作品仓库：https://gitcode.com/sun-yongji-yuyubenyuan_admin/taichi-matrix
