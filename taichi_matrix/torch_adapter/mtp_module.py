"""
TaiChi MTP Module — Multi-Token Prediction PyTorch wrapper.

Wraps the six-yao depth-scheduled multi-token prediction of
taichi-mtp.  Six prediction heads are coupled through a C6
coupling matrix derived from the cyclic group rotations.

Mathematical basis
------------------
The C6 coupling matrix C is constructed so that C[i][j] = exp(i·j·2π/6),
forming a 6×6 unitary matrix.  Each prediction head h_k projects
hidden states to the vocabulary, then the coupling matrix mixes them:

    ŷ = C · H · W_out

Depth scheduling follows the I Ching six-yao positions:
  • Shallow  (yao 0-1): 2 heads active
  • Medium   (yao 2-4): 4 heads active
  • Deep     (yao 5):    all 6 heads active
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from taichi_matrix.constants import (
    C6_ORDER,
    C6_ROTATION_ANGLE,
    GOLDEN_RATIO_COMPENSATION,
)


@dataclass
class MTPResult:
    """Output of :class:`TaiChiMTPModule`."""

    #: Predicted logits per token, shape ``(B, 6, V)``.
    predictions: torch.Tensor
    #: Depth mode used: "shallow", "medium", or "deep".
    depth_mode: str
    #: Coupling matrix applied, shape ``(6, 6)``.
    coupling_matrix: torch.Tensor
    #: Per-token confidence scores, shape ``(B, 6)``.
    confidence: torch.Tensor


class TaiChiMTPModule(nn.Module):
    """C6-coupled Multi-Token Prediction module.

    Parameters
    ----------
    hidden_dim : int
        Input hidden dimension.
    vocab_size : int
        Output vocabulary size.
    num_heads : int
        Number of prediction heads (default 6, C6_ORDER).
    coupling_strength : float
        Strength of the C6 coupling mixing (0 = independent heads).
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        vocab_size: int = 32000,
        num_heads: int = C6_ORDER,
        coupling_strength: float = 1.0,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.num_heads = num_heads
        self.coupling_strength = coupling_strength

        # 6 independent prediction heads
        self.heads = nn.ModuleList(
            [nn.Linear(hidden_dim, vocab_size, bias=False) for _ in range(num_heads)]
        )

        # Depth scheduler: projection to decide depth mode
        self.depth_classifier = nn.Sequential(
            nn.Linear(hidden_dim, 16),
            nn.GELU(),
            nn.Linear(16, 3),  # 3 classes: shallow/medium/deep
        )

        # Confidence head
        self.confidence_head = nn.Linear(hidden_dim, num_heads)

        self._build_coupling_matrix()
        self._reset_parameters()

    def _build_coupling_matrix(self) -> None:
        """Build the C6 unitary coupling matrix.

        C[i][j] = exp(2πi · i·j / 6) — DFT of the cyclic group.
        """
        import numpy as np
        c = np.zeros((self.num_heads, self.num_heads), dtype=complex)
        for i in range(self.num_heads):
            for j in range(self.num_heads):
                angle = 2.0 * np.pi * i * j / self.num_heads
                c[i, j] = np.exp(1j * angle)
        # Store as real 2-channel (cos, sin) for pure torch computation
        self.register_buffer(
            "coupling_cos", torch.tensor(c.real, dtype=torch.float32)
        )
        self.register_buffer(
            "coupling_sin", torch.tensor(c.imag, dtype=torch.float32)
        )

    def _reset_parameters(self) -> None:
        for p in self.parameters():
            if p.dim() >= 2:
                nn.init.xavier_uniform_(p)

    # ------------------------------------------------------------------
    # Depth scheduler
    # ------------------------------------------------------------------

    @staticmethod
    def _get_active_heads(depth_mode: str, num_heads: int = 6) -> list[int]:
        """Return indices of heads active for a given depth mode.

        Shallow: heads 0-1, Medium: heads 0-3, Deep: all 6.
        """
        if depth_mode == "shallow":
            return list(range(min(2, num_heads)))
        elif depth_mode == "medium":
            return list(range(min(4, num_heads)))
        return list(range(num_heads))

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        hidden_states: torch.Tensor,
        target_depth: Optional[str] = None,
    ) -> MTPResult:
        """Generate multi-token predictions with C6 coupling.

        Parameters
        ----------
        hidden_states : Tensor, shape ``(B, D)`` or ``(D,)``.
        target_depth : str, optional
            Override depth mode. If None, auto-detected.

        Returns
        -------
        MTPResult with predictions, depth_mode, coupling_matrix, confidence.
        """
        if hidden_states.dim() == 1:
            hidden_states = hidden_states.unsqueeze(0)
        B = hidden_states.shape[0]

        # 1. Determine depth
        if target_depth is None:
            logits_3 = self.depth_classifier(hidden_states)  # (B, 3)
            mode_idx = logits_3.argmax(dim=-1)
            depth_mode = ["shallow", "medium", "deep"][mode_idx[0].item()]
        else:
            depth_mode = target_depth

        active = self._get_active_heads(depth_mode, self.num_heads)

        # 2. Run active heads
        head_outs = []
        for k, head in enumerate(self.heads):
            out_k = head(hidden_states)  # (B, V)
            if k in active:
                head_outs.append(out_k)
            else:
                head_outs.append(torch.zeros_like(out_k))

        # Stack: (B, num_heads, V)
        stacked = torch.stack(head_outs, dim=1)

        # 3. Apply C6 coupling (complex multiply via cos/sin)
        # coupling_cos/sin: (H, H), stacked: (B, H, V)
        # Use einsum to mix heads: result[b,i,V] = sum_j C[i,j] * stacked[b,j,V]
        coupled_real = torch.einsum('ij,bjV->biV', self.coupling_cos, stacked)
        coupled_imag = torch.einsum('ij,bjV->biV', self.coupling_sin, stacked)
        # Weight by coupling strength and combine
        coupled = (
            coupled_real * self.coupling_strength
            + coupled_imag * self.coupling_strength
        )
        # Project back to vocab size (truncate if needed)
        predictions = coupled[..., :self.vocab_size]

        # 4. Confidence scores
        confidence = torch.sigmoid(self.confidence_head(hidden_states))

        # 5. Coupling matrix (real part for reporting)
        c_matrix = self.coupling_cos * self.coupling_strength

        return MTPResult(
            predictions=predictions,
            depth_mode=depth_mode,
            coupling_matrix=c_matrix,
            confidence=confidence,
        )

    def extra_repr(self) -> str:
        return (
            f"hidden_dim={self.hidden_dim}, vocab_size={self.vocab_size}, "
            f"num_heads={self.num_heads}, coupling_strength={self.coupling_strength}"
        )
