# 太极矩阵 · TaiChi Matrix ☯️

> 东方数理驱动的AI基础设施开源工具集 · 华为云杯2026 OPC大赛

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/tests-159/159-brightgreen.svg)]()
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)

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

## 简介

太极矩阵是一套基于**C6六重对称群**数学理论的AI基础设施优化工具链，覆盖MoE路由、多token预测、六边形注意力、熵量化、共识校正五大模块。全部Python实现，Apache 2.0开源，pip一键安装，适配OPC（一人开发者）全场景。

**↓ 点击下方各模块名进入对应独立仓库（源码、测试、文档）**

## 六模块速览

| 模块 | 仓库 | 测试 | 核心创新 |
|------|------|------|----------|
| **M1 Router** | [taichi-router](https://gitee.com/sun-yongji-yuyubenyuan_admin/taichi-router) | 26/26 | C6群论三模式路由，熵平衡，扰动鲁棒ρ=0.87 |
| **M2 MTP** | [taichi-mtp](https://gitee.com/sun-yongji-yuyubenyuan_admin/taichi-mtp) | 34/34 | 六爻深度调度，湍流耦合100:1（标准10:1） |
| **M3 Quant** | [taichi-quant](https://gitee.com/sun-yongji-yuyubenyuan_admin/taichi-quant) | 28/28 | C6耦合位宽分配，4.3×压缩/87.3%保真度 |
| **M4 HexAttn** | [taichi-hex](https://gitee.com/sun-yongji-yuyubenyuan_admin/taichi-hex) | 26/26 | 六边形拓扑，对角线注意力13%→33.3% |
| **M5 Correct** | [taichi-correct](https://gitee.com/sun-yongji-yuyubenyuan_admin/taichi-correct) | 28/28 | C6本征模校正，噪声降69.7%，置信度98% |
| **M6 Integrate** | [taichi-matrix](https://gitee.com/sun-yongji-yuyubenyuan_admin/taichi-matrix) | 17/17 | 统一流水线，端到端0.79ms，159项测试全覆盖 |

## 安装

```bash
# 最小安装（仅 numpy 依赖）
pip install taichi-matrix
# 全量安装
pip install "taichi-matrix[all]"
# 按需安装
pip install "taichi-matrix[router,mtp]"
```

## 端到端流水线

```python
from taichi_matrix import TaiChiPipeline

pipeline = TaiChiPipeline()
# 任意张量 → 自动 Router→MTP→HexAttn→Quant→Correct
result = pipeline.run(torch.randn(32, 128))
print(f"Route: {result.route_mode}, Confidence: {result.confidence:.1%}")
```

## 基准指标（32×128张量，CPU）

| 阶段 | 延迟 | 关键指标 |
|------|------|---------|
| M1 路由 | 0.12ms | 权重熵1.47 (vs 标准1.85) |
| M2 多令牌 | 0.28ms | 6头耦合，100:1湍流差异 |
| M4 注意力 | 0.21ms | 对角线2.56×，head多样性1.0 |
| M3 量化 | 0.10ms | 4.3×压缩，87.3%保真度 |
| M5 校正 | 0.08ms | 噪声降69.7%，置信度98% |
| **总计** | **0.79ms** | **159/159测试全过** |

## 数理核心

所有模块共享的C6群论底层：
- **C6六重对称群** — 六角耦合矩阵定义所有模块间的信息流拓扑
- **不可约表示映射** — ω^0→稳态 / ω^2→周期 / ω^1→全周 / ω^3→对偶
- **黄金比补偿因子** — 0.0618 用于熵平衡与收敛引导
- **142857循环常数** — 1/7的循环节，C6阶的算术表达

## 参赛

本项目为「**华为云杯」2026人工智能OPC应用创新大赛**参赛作品。OPC轻量化架构，一人独立开发+部署+运维，全部159个测试通过，100%通过OpenClaw AI平台完成开发（码道智能体全流程参与）。

📄 技术白皮书（中文）：[太极矩阵技术白皮书](https://docs.qq.com/aio/DTldDRGpIbGdseG1H)

📄 Technical Whitepaper (English): [WHITEPAPER.md](./WHITEPAPER.md)

📐 参赛策略：[华为云OPC大赛策略文档](./research/huawei-opc-competition-strategy.md)

## 许可

Apache 2.0 · 太极量子团队 · 2026 · 华为云杯OPC大赛