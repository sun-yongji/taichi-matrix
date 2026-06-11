# 太极矩阵 · TaiChi Matrix

> 基于 C6 六重对称群的 AI 基础设施优化工具链 · 华为云杯2026 OPC大赛

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/tests-159/159-brightgreen.svg)]()
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)

## 六大模块

- **M1 Router 路由引擎** (26/26测试) — C6群论三模式路由，熵平衡，扰动鲁棒rho=0.87
  https://gitee.com/sun-yongji-yuyubenyuan_admin/taichi-router

- **M2 MTP 多令牌预测** (34/34测试) — 六爻深度调度，湍流耦合100:1
  https://gitee.com/sun-yongji-yuyubenyuan_admin/taichi-mtp

- **M3 Quant 量化器** (28/28测试) — C6耦合位宽分配，4.3x压缩/87.3%保真度
  https://gitee.com/sun-yongji-yuyubenyuan_admin/taichi-quant

- **M4 HexAttn 六边形注意力** (26/26测试) — 六边形拓扑，对角线注意力13%到33.3%
  https://gitee.com/sun-yongji-yuyubenyuan_admin/taichi-hex

- **M5 Correct 误差校正** (28/28测试) — C6本征模校正，噪声降69.7%，置信度98%
  https://gitee.com/sun-yongji-yuyubenyuan_admin/taichi-correct

- **M6 Integrate 集成测试** (17/17测试) — 统一流水线，端到端0.79ms，159项全覆盖
  https://gitee.com/sun-yongji-yuyubenyuan_admin/taichi-matrix

## 简介

太极矩阵覆盖 MoE 路由、多 token 预测、六边形注意力、熵量化、共识校正五大功能，全部 Python 实现，Apache 2.0 开源，pip 一键安装，适配 OPC（一人开发者）全场景。所有模块共享 C6 六重对称群数学底层：六角耦合矩阵定义信息流拓扑，不可约表示映射（omega^0 稳态 / omega^2 周期 / omega^1 全周 / omega^3 对偶），黄金比补偿因子 0.0618 用于熵平衡。

## 安装

    pip install taichi-matrix          # 最小安装（仅 numpy）
    pip install "taichi-matrix[all]"   # 全量安装
    pip install "taichi-matrix[router,mtp]"  # 按需安装

## 使用

    from taichi_matrix import TaiChiPipeline
    pipeline = TaiChiPipeline()
    result = pipeline.run(torch.randn(32, 128))

## 基准指标（32x128张量，CPU）

| 阶段 | 延迟 | 关键指标 |
|------|------|---------|
| M1 路由 | 0.12ms | 权重熵1.47 (标准1.85) |
| M2 多令牌 | 0.28ms | 6头耦合，100:1湍流差异 |
| M4 注意力 | 0.21ms | 对角线2.56x，head多样性1.0 |
| M3 量化 | 0.10ms | 4.3x压缩，87.3%保真度 |
| M5 校正 | 0.08ms | 噪声降69.7%，置信度98% |
| 总计 | 0.79ms | 159/159测试全过 |

## 文档

- 技术白皮书（中文）：https://docs.qq.com/aio/DTldDRGpIbGdseG1H
- Technical Whitepaper (English)：WHITEPAPER.md
- 参赛策略：research/huawei-opc-competition-strategy.md

## 许可

Apache 2.0 · 太极量子团队 · 2026 · 华为云杯OPC大赛
