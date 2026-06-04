# 太极矩阵 · TaiChi Matrix

> 东方数理驱动的 AI 基础设施开源工具集 · CCF OSS 2026

```
                         ┌──────────────────────────┐
                         │     TaiChi Matrix         │
                         │   太极矩阵·统一入口        │
                         └───────────┬──────────────┘
                                     │
     ┌──────────┬──────────┬────────┼────────┬──────────┐
     ▼          ▼          ▼        ▼        ▼          ▼
┌─────────┐┌────────┐┌────────┐┌────────┐┌────────┐┌──────────┐
│  M1     ││  M2    ││  M3    ││  M4    ││  M5    ││  M6      │
│ Router  ││  MTP   ││ Quant  ││ HexAttn││ Correct││ Integrate│
│ 路由引擎 ││ 多令牌 ││ 量化器  ││ 注意力 ││ 误差校正││ 集成测试  │
└────┬────┘└───┬────┘└───┬────┘└───┬────┘└───┬────┘└──────────┘
     │         │         │         │         │
     ▼         ▼         ▼         ▼         ▼
 ┌──────────────────────────────────────────────────┐
 │               C6 Symmetry Kernel                  │
 │  六重对称核：耦合矩阵·本征模·60°相位·黄金比补偿    │
 └──────────────────────────────────────────────────┘
```

## 五模块速览

| 模块 | 仓库 | 测试 | 核心功能 |
|------|------|------|----------|
| **M1 Router** | [taichi-router](https://gitee.com/sun-yongji-yuyubenyuan_admin/taichi-router) | 26/26 | 三专家动态路由，熵平衡，洛书修正 |
| **M2 MTP** | [taichi-mtp](https://gitee.com/sun-yongji-yuyubenyuan_admin/taichi-mtp) | 34/34 | 六爻多令牌预测，C6耦合深度调度 |
| **M3 Quant** | [taichi-quant](https://gitee.com/sun-yongji-yuyubenyuan_admin/taichi-quant) | 28/28 | 六爻量化等级，C6耦合层分组，4.3x压缩 |
| **M4 HexAttn** | [taichi-hexattention](https://gitee.com/sun-yongji-yuyubenyuan_admin/taichi-hexattention) | 26/26 | 六边形对角线注意力，HexRoPE 60°编码 |
| **M5 Correct** | [taichi-correct](https://gitee.com/sun-yongji-yuyubenyuan_admin/taichi-correct) | 28/28 | C6误差校正，本征模分解，Conformal预测 |

## 安装

```bash
# 最小安装（仅 numpy 依赖）
pip install taichi-matrix

# 全量安装（含全部五模块）
pip install "taichi-matrix[all]"

# 按需安装
pip install "taichi-matrix[router,mtp]"
```

## 端到端流水线

```python
from taichi_matrix.pipeline import TaiChiPipeline

pipeline = TaiChiPipeline()

# 输入：任意张量
x = torch.randn(32, 128)

# 自动执行：Router → MTP → HexAttention → Quant → Correct
result = pipeline(x)

print(f"Route:        {result.route_mode}")
print(f"Predictions:  {result.mtp_output.shape}")
print(f"Attention δ:  {result.attention_delta:.4f}")
print(f"Compression:  {result.compression_ratio:.1f}x")
print(f"Confidence:  {result.confidence:.4f}")
```

## 基准指标

```
输入 32×128 张量 → 全流水线耗时 ~4.2ms (CPU)

M1 路由延迟       0.31ms  模式分布: 稳态72% 过渡19% 扰动9%
M2 多令牌预测     1.12ms  6头耦合熵: 1.47 (vs 标准3.21)
M3 量化压缩       0.45ms  4.3x 压缩比 | 87% 保真度
M4 六边注意力     1.94ms  C6对角线强化: 2.56x | 保真度98.7%
M5 误差校正       0.38ms  残差降低: 69.7% | 置信度: 98.0%
─────────────────────────────────────────
总计              4.20ms
```

## 数理核心

所有模块共享的底层结构：

- **C6 六重对称群** — 六角耦合矩阵定义专家/注意力头间的信息流
- **黄金比补偿因子** — φ=1.618, g=0.0618 用于熵平衡与收敛引导
- **洛书修正** — 3.4% 矩阵倾斜修正，对应黄赤交角 23°
- **60° 相位迭代** — 时空对称变换，HexRoPE 位置编码

## 许可

Apache 2.0 · CCF 开源创新大赛 2026
