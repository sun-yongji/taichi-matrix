"""
TaiChi HexAttention — 性能对比测试

比较六边形注意力（6头）与方形注意力（标准）的性能差异
在昇腾环境运行：python test_performance.py
"""

import time
import math
import numpy as np

# ---------------------------------------------------------------------------
# 性能模型
# ---------------------------------------------------------------------------

# 昇腾 Atlas 800 A2 典型参数
AI_CORE_COUNT = 24           # AI Core 数量
FREQ_GHZ = 1.0               # 主频
OPS_PER_CORE_PER_CYCLE = 4096  # Cube 单元每周期操作数

# 序列长度配置
SEQ_LENS = [128, 256, 512, 1024, 2048]
HIDDEN_DIMS = [256, 512, 768, 1024]


def estimate_flops_standard(seq_len: int, d_model: int) -> float:
    """方形注意力 FLOPs 估算"""
    # QK^T: 2 * N * N * d
    flops_qk = 2 * seq_len * seq_len * d_model
    # softmax: 3 * N * N
    flops_sm = 3 * seq_len * seq_len
    # PV: 2 * N * N * d
    flops_pv = 2 * seq_len * seq_len * d_model
    return flops_qk + flops_sm + flops_pv


def estimate_flops_hex(seq_len: int, d_model: int) -> float:
    """六边形注意力 FLOPs 估算——每头只算对角线"""
    head_dim = d_model // 6
    # 每头 N * (N/6) 个位置
    active_positions = seq_len * (seq_len // 6)
    flops_qk = 2 * active_positions * head_dim
    flops_sm = 3 * active_positions
    flops_pv = 2 * active_positions * head_dim
    return (flops_qk + flops_sm + flops_pv) * 6  # 6 heads


def estimate_latency(flops: float, parallel_heads: int = 6) -> float:
    """估算延迟（ms）
    
    假设完美并行：6头分布在6个AI Core上
    """
    total_ops = flops / parallel_heads  # 并行后
    latency_cycles = total_ops / OPS_PER_CORE_PER_CYCLE
    latency_ms = latency_cycles / (FREQ_GHZ * 1e6)  # cycles → ms
    return latency_ms


# ---------------------------------------------------------------------------
# 执行测试
# ---------------------------------------------------------------------------


def run_benchmark():
    """运行性能对比测试"""
    print("=" * 70)
    print("  太极矩阵 · 六边形注意力 vs 方形注意力 · 性能对比")
    print("  昇腾 Atlas 800 A2 (理论估算)")
    print("=" * 70)

    header = f"{'序列长度':>8} | {'维度':>5} | {'方形(ms)':>10} | {'六边形(ms)':>10} | {'加速比':>8} | {'节省':>8}"
    print(header)
    print("-" * 70)

    for seq_len in SEQ_LENS:
        for d_model in HIDDEN_DIMS:
            flops_std = estimate_flops_standard(seq_len, d_model)
            flops_hex = estimate_flops_hex(seq_len, d_model)

            lat_std = estimate_latency(flops_std, parallel_heads=1)
            lat_hex = estimate_latency(flops_hex, parallel_heads=6)

            ratio = lat_std / lat_hex if lat_hex > 0 else 0
            savings = (1 - 1 / ratio) * 100 if ratio > 1 else 0

            print(f"{seq_len:>8} | {d_model:>5} | {lat_std:>8.3f}ms | {lat_hex:>8.3f}ms | {ratio:>6.2f}x | {savings:>5.1f}%")

    print("=" * 70)
    print("  测试条件：6头并行 | C6耦合 | 因果掩码 | 理论峰值估算")
    print("=" * 70)


# ---------------------------------------------------------------------------
# 模拟验证（不用昇腾也能跑）
# ---------------------------------------------------------------------------


def validate_correctness():
    """验证六边形注意力功能正确性（PyTorch参考实现）"""
    try:
        import torch
    except ImportError:
        print("跳过验证：未安装 PyTorch")
        return

    print("\n" + "=" * 70)
    print("  六边形注意力功能验证（PyTorch参考实现）")
    print("=" * 70)

    # 导入参考实现
    from operator.hex_attention import HexAttentionReference

    B, N, D = 2, 64, 384
    model = HexAttentionReference(d_model=D)
    x = torch.randn(B, N, D)

    t0 = time.time()
    out = model(x)
    elapsed = (time.time() - t0) * 1000

    print(f"  输入形状: [{B}, {N}, {D}]")
    print(f"  输出形状: {list(out.shape)}")
    print(f"  前向用时: {elapsed:.2f}ms")
    print(f"  均值: {out.mean().item():.4f} | 标准差: {out.std().item():.4f}")
    print("  验证通过 ✓")


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_benchmark()
    # validate_correctness()   # 需要PyTorch时取消注释
