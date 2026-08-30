"""
TaiChi HexAttention — Ascend CANN Custom Operator
==================================================

基于C6六重对称群的六边形注意力算子
6方向映射 · 6头并行 · 60°六边形拓扑

Input:  x [B, N, D]  — batch, seq_len, hidden_dim
Output: y [B, N, D]  — attended output

CANN版本要求: 8.5.0+
操作系统要求: EulerOS 2.10
"""

import math
from functools import reduce

# ============================================================================
# CANN TBE DSL (昇腾算子开发框架)
# ============================================================================

try:
    # CANN 8.5.0 TBE DSL
    import te.lang.cce as tbe
    from te import tvm
    from te.platform import CUBE_MKN
    from te.utils import para_check, shape_relation
    from topi import generic
    from topi.cce import util
    HAS_CANN = True
except ImportError:
    HAS_CANN = False

# ============================================================================
# C6 六重对称群常量
# ============================================================================

NUM_HEADS = 6                      # 6个方向 → 6个注意力头
HEX_ANGLE = math.pi / 3.0         # 60° = C6旋转角
PHI_COMP = 0.0618                 # 黄金比补偿因子

HEAD_NAMES = ["Head-0 (0°)", "Head-1 (60°)", "Head-2 (120°)", "Head-3 (180°)", "Head-4 (240°)", "Head-5 (300°)"]

# C6耦合矩阵 — 六边形拓扑
# 每个头的输出按此矩阵与其他头混合
C6_COUPLING = [
    [1.000, 0.500, 0.000, 0.000, 0.000, 0.500],
    [0.500, 1.000, 0.500, 0.000, 0.000, 0.000],
    [0.000, 0.500, 1.000, 0.500, 0.000, 0.000],
    [0.000, 0.000, 0.500, 1.000, 0.500, 0.000],
    [0.000, 0.000, 0.000, 0.500, 1.000, 0.500],
    [0.500, 0.000, 0.000, 0.000, 0.500, 1.000],
]

# ============================================================================
# 六边形注意力掩码生成
# ============================================================================


@util.check_input_type((list,), (int,), (bool,))
def gen_hexagonal_mask(shape, head_id, causal=True):
    """生成六边形对角注意力掩码

    头head_id关注 (q_pos - k_pos) % 6 == head_id 的位置
    6个头合起来覆盖所有位置对，每头只算1/6

    Args:
        shape: [B, N, D] shape 信息
        head_id: 注意力头编号 [0,5] 对应 Head-0→Head-5
        causal: 因果掩码（只看过去位置）

    Returns:
        注意力掩码 [N, N]，1=关注，0=掩码
    """
    seq_len = shape[-2]
    mask = []
    for q in range(seq_len):
        row = []
        for k in range(seq_len):
            if causal and k > q:
                row.append(0)
            elif (k - q) % NUM_HEADS == head_id:
                row.append(1)
            else:
                row.append(0)
        mask.append(row)
    return mask


def gen_all_hex_masks(shape, causal=True):
    """生成全部6个头的掩码"""
    return [gen_hexagonal_mask(shape, h, causal) for h in range(NUM_HEADS)]


# ============================================================================
# 六边形注意力算子主实现
# ============================================================================


def hex_attention_compute(query, key, value, head_id, scale, kernel_name="hex_attn"):
    """单个六边形注意力头的计算

    每个头只计算一条对角线上的注意力，
    约 full attention 的 1/6 计算量。

    Args:
        query: [B, N, D] 查询
        key:   [B, N, D] 键
        value: [B, N, D] 值
        head_id: 头编号 [0,5]
        scale: 缩放因子 1/sqrt(head_dim)
        kernel_name: 算子名称

    Returns:
        该头输出 [B, N, D]
    """
    # 1. 生成掩码（实际TBE实现中为compile-time常量）
    mask = gen_hexagonal_mask(query.shape, head_id)

    # 2. 注意力分数 Q @ K.T / scale
    #    TBE: tbe.matmul(query, key, trans_b=True)
    scores = tbe.matmul(query, key, trans_b=True)
    scores = tbe.vmuls(scores, scale)

    # 3. 六边形掩码
    mask_tensor = tbe.broadcast(
        tvm.const(mask, dtype=scores.dtype),
        scores.shape
    )
    # 掩码位置设为 -inf
    scores = tbe.vadd(scores, tbe.vmuls(mask_tensor, -1e9))

    # 4. softmax
    weights = tbe.softmax(scores)

    # 5. 加权求和 weights @ value
    output = tbe.matmul(weights, value)

    return output


# ============================================================================
# 算子入口 — 6头计算 + C6耦合
# ============================================================================


# 算子输入校验
@util.check_input_type(
    (tvm.tensor.Tensor,),  # query  [B, N, D]
    (tvm.tensor.Tensor,),  # key    [B, N, D]
    (tvm.tensor.Tensor,),  # value  [B, N, D]
    (tvm.tensor.Tensor,),  # output [B, N, D]
    (str,),                 # kernel_name
)
def hex_attention_ascend(query, key, value, output, kernel_name="hex_attention"):
    """六边形注意力昇腾算子

    C6六重对称群映射为6头并行注意力，
    每头覆盖不同对角线方向，C6耦合矩阵混合输出。

    Args:
        query: [B, N, D] 查询张量
        key:   [B, N, D] 键张量
        value: [B, N, D] 值张量
        output: [B, N, D] 输出张量
        kernel_name: 算子名称（默认 hex_attention）
    """
    if not HAS_CANN:
        raise RuntimeError(
            "CANN TBE required. "
            "请在 Euler 2.10 + CANN 8.5.0 环境中运行"
        )

    # 获取输入维度
    shape = query.shape
    batch, seq_len, d_model = shape[0], shape[1], shape[2]
    head_dim = d_model // NUM_HEADS
    scale = 1.0 / math.sqrt(head_dim)

    # 6头并行计算
    head_outputs = []
    for h in range(NUM_HEADS):
        head_out = hex_attention_compute(
            query, key, value, h, scale,
            f"{kernel_name}_head_{h}"
        )
        head_outputs.append(head_out)

    # C6耦合混合: 6个头输出按C6耦合矩阵加权
    # output = sum(head_outputs[h] * C6_COUPLING[h][:]) for h in 0..5
    coupled = None
    for h in range(NUM_HEADS):
        weighted = tbe.vmuls(head_outputs[h], C6_COUPLING[0][h])
        if coupled is None:
            coupled = weighted
        else:
            coupled = tbe.vadd(coupled, weighted)

    # 输出赋值
    tbe.vadd(output, coupled)

    # 构建算子
    with tvm.target.cce():
        sch = generic.auto_schedule(output)
    config = {
        "kernel_name": kernel_name,
        "need_build": True,
        "need_print": False,
    }
    tbe.build(sch, config)


# ============================================================================
# PyTorch参考实现（用于精度验证，非昇腾环境）
# ============================================================================

class HexAttentionTorch(torch.nn.Module):
    """六边形注意力 PyTorch 参考实现

    在昇腾环境外验证算法正确性使用
    """

    def __init__(self, d_model=512):
        super().__init__()
        self.d_model = d_model
        self.num_heads = NUM_HEADS
        self.head_dim = d_model // NUM_HEADS
        self.scale = self.head_dim ** -0.5

        self.q_proj = torch.nn.Linear(d_model, d_model, bias=False)
        self.k_proj = torch.nn.Linear(d_model, d_model, bias=False)
        self.v_proj = torch.nn.Linear(d_model, d_model, bias=False)
        self.out_proj = torch.nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        B, N, D = x.shape
        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        head_dim = D // self.num_heads

        head_outs = []
        for h in range(self.num_heads):
            # 6头独立六边形注意力
            q_h = q[:, :, h*head_dim:(h+1)*head_dim]
            k_h = k[:, :, h*head_dim:(h+1)*head_dim]
            v_h = v[:, :, h*head_dim:(h+1)*head_dim]

            scores = torch.matmul(q_h, k_h.transpose(-2, -1)) * self.scale

            # 六边形掩码
            mask = torch.zeros(N, N, device=x.device)
            for qi in range(N):
                for ki in range(N):
                    if ki > qi:  # causal
                        mask[qi, ki] = float("-inf")
                    elif (ki - qi) % self.num_heads == h:
                        mask[qi, ki] = 0.0
                    else:
                        mask[qi, ki] = float("-inf")

            scores = scores + mask.unsqueeze(0)
            weights = torch.softmax(scores, dim=-1)
            out_h = torch.matmul(weights, v_h)
            head_outs.append(out_h)

        # C6耦合混合
        stacked = torch.stack(head_outs, dim=-1)
        coupling = torch.tensor(C6_COUPLING, dtype=torch.float32, device=x.device)
        mixed = torch.einsum("bndh,hi->bndi", stacked, coupling)
        out = mixed.mean(dim=-1)

        return self.out_proj(out)
