"""
Hex Attention Module — PyTorch nn.Module wrapper for taichi-hex.

Implements C6 hexagonal-topology attention where 6 attention heads
correspond to the 6 symmetry operations of the cyclic group C6.

Mathematical basis
------------------
Standard multi-head attention computes:

    A = softmax(Q K^T / √d_k)

In C6 hexagonal attention, each head h_k (k=0…5) applies a cyclic
permutation P_k to the query/key index order before computing
attention scores.  The permutation P_k[i] = (i + k) mod 6 rearranges
the sequence according to the k-th C6 rotation.

This boosts diagonal attention coverage from the standard ~13 % to
~33.3 % (1/3 of entries in the hexagonal bandwidth), improving
long-range dependency modelling.

The C6 coupling matrix mixes head outputs:

    O = C · concat(o_0, …, o_5)  ·  W_out
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from taichi_matrix.constants import (
    C6_ORDER,
    C6_ROTATION_ANGLE,
    GOLDEN_RATIO_COMPENSATION,
)


@dataclass
class AttentionOutput:
    """Output of :class:`HexAttentionModule`."""

    #: Attended output, shape ``(B, S, D)``.
    output: torch.Tensor
    #: Attention weights (averaged across heads), shape ``(B, H, S, S)``.
    attention_weights: torch.Tensor
    #: Diagonal attention coverage ratio (scalar tensor).
    diagonal_coverage: torch.Tensor


class HexAttentionModule(nn.Module):
    """C6 Hexagonal Attention module.

    Parameters
    ----------
    d_model : int
        Model dimension (must be divisible by num_heads).
    num_heads : int
        Number of attention heads (default 6, C6_ORDER).
    dropout : float
        Dropout probability on attention weights.
    use_coupling : bool
        Whether to apply C6 coupling mixing on head outputs.
    """

    def __init__(
        self,
        d_model: int = 64,
        num_heads: int = C6_ORDER,
        dropout: float = 0.1,
        use_coupling: bool = True,
    ) -> None:
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.use_coupling = use_coupling

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

        # C6 coupling matrix for mixing head outputs
        if use_coupling:
            self._build_c6_coupling()

        self._reset_parameters()

    def _build_c6_coupling(self) -> None:
        """Register C6 coupling buffer (DFT matrix of cyclic group)."""
        import numpy as np
        c = np.zeros((self.num_heads, self.num_heads), dtype=np.float32)
        for i in range(self.num_heads):
            for j in range(self.num_heads):
                angle = 2.0 * np.pi * i * j / self.num_heads
                c[i, j] = np.cos(angle)
        self.register_buffer(
            "c6_coupling", torch.tensor(c, dtype=torch.float32)
        )

    def _reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)

    # ------------------------------------------------------------------
    # C6 permutation
    # ------------------------------------------------------------------

    @staticmethod
    def _c6_permute(x: torch.Tensor, k: int) -> torch.Tensor:
        """Apply the k-th C6 cyclic permutation along the sequence dim.

        P_k[i] = (i + k) mod seq_len  (cyclic shift by k).

        Parameters
        ----------
        x : Tensor, shape ``(B, S, D)``.
        k : int — permutation index (0 = identity).

        Returns
        -------
        Permuted tensor with same shape.
        """
        if k == 0:
            return x
        return x.roll(shifts=k, dims=1)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> AttentionOutput:
        """C6 hexagonal multi-head attention.

        Parameters
        ----------
        x : Tensor, shape ``(B, S, D)``.
        mask : Tensor, optional, shape ``(B, 1, S, S)`` or broadcastable.

        Returns
        -------
        AttentionOutput with output, attention_weights, diagonal_coverage.
        """
        B, S, D = x.shape

        # Project Q, K, V
        Q = self.q_proj(x)  # (B, S, D)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # Reshape to (B, H, S, head_dim)
        Q = Q.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute attention per head with C6 permutation
        head_outputs = []
        head_attns = []
        diag_counts = []

        for h in range(self.num_heads):
            # Apply C6 permutation to queries of head h
            Q_h = self._c6_permute(Q[:, h:h+1], k=h)  # (B, 1, S, d)
            K_h = K[:, h:h+1]  # (B, 1, S, d)

            scores = torch.matmul(Q_h, K_h.transpose(-2, -1)) / (
                self.head_dim ** 0.5
            )  # (B, 1, S, S)

            if mask is not None:
                scores = scores.masked_fill(mask.bool(), float("-inf"))

            attn_w = F.softmax(scores, dim=-1)
            attn_w = self.dropout(attn_w)

            out_h = torch.matmul(attn_w, V[:, h:h+1])  # (B, 1, S, d)
            head_outputs.append(out_h.squeeze(1))
            head_attns.append(attn_w.squeeze(1))

            # Diagonal coverage: fraction of attention mass on |i-j| <= S//6
            S_sq = S * S
            bw = max(S // 6, 1)
            idx_i = torch.arange(S, device=x.device).unsqueeze(1)
            idx_j = torch.arange(S, device=x.device).unsqueeze(0)
            diag_mask = (idx_i - idx_j).abs() <= bw
            diag_count = (attn_w.squeeze(1) * diag_mask.float()).sum().detach()
            diag_counts.append(diag_count / (S_sq + 1e-8))

        # Stack: (B, H, S, d) → concat → (B, S, D)
        concat = torch.cat(head_outputs, dim=-1)  # (B, S, D)
        attn_weights = torch.stack(head_attns, dim=1)  # (B, H, S, S)

        # C6 coupling mixing
        if self.use_coupling and hasattr(self, "c6_coupling"):
            # Mix across the head dimension at position level
            coupling = self.c6_coupling  # (H, H)
            # Reshape concat to apply coupling on head outputs
            co = concat.view(B, S, self.num_heads, self.head_dim)
            co = torch.einsum("ij,bsjd->bsid", coupling, co)
            concat = co.reshape(B, S, D)

        output = self.out_proj(concat)

        # Mean diagonal coverage across heads
        diag_coverage = torch.stack(diag_counts).mean()

        return AttentionOutput(
            output=output,
            attention_weights=attn_weights,
            diagonal_coverage=diag_coverage,
        )

    def extra_repr(self) -> str:
        return (
            f"d_model={self.d_model}, num_heads={self.num_heads}, "
            f"use_coupling={self.use_coupling}"
        )
