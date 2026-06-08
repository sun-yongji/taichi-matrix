"""
TaiChi Quant Module — Quantization PyTorch wrapper.

Wraps taichi-quant's C6-coupled bit-width allocation and
symmetric linear quantization as a PyTorch nn.Module.

Mathematical basis
------------------
C6 coupling distributes 6 bit-width options {4, 5, 6, 7, 8, 16} across
layers following the cyclic group permutation.  For each layer the
quantisation step is:

    scale = max(|w|) / (2^(b-1) - 1)
    q_w = round(w / scale) · scale

A ``fake_quantize`` method (straight-through estimator) is provided
so that the module can be used inside training graphs while the
underlying weights remain in full precision.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from taichi_matrix.constants import (
    C6_ORDER,
    CYCLIC_CONSTANT_142857,
    GOLDEN_RATIO_COMPENSATION,
)


@dataclass
class QuantResult:
    """Output of :class:`TaiChiQuantModule`."""

    #: Compressed (quantised) weights, same shape as input, detached.
    compressed: torch.Tensor
    #: Recovered (dequantised) weights, same shape, differentiable.
    recovered: torch.Tensor
    #: Per-layer bit widths used, shape ``(B,)``.
    bit_widths: torch.Tensor
    #: Dictionary of compression metrics.
    metrics: Dict[str, float]


class TaiChiQuantModule(nn.Module):
    """C6-coupled symmetric quantisation module.

    Parameters
    ----------
    base_bit_width : int
        Default bit width (must be 4–8 or 16).
    num_bit_options : int
        Number of bit-width choices (default 6, matching C6_ORDER).
    coupling_strength : float
        How strongly C6 cyclic coupling influences bit allocation.
    """

    # Bit-width candidates arranged in C6-cyclic order
    BIT_OPTIONS: List[int] = [4, 5, 6, 7, 8, 16]

    def __init__(
        self,
        base_bit_width: int = 8,
        num_bit_options: int = C6_ORDER,
        coupling_strength: float = GOLDEN_RATIO_COMPENSATION,
    ) -> None:
        super().__init__()
        self.base_bit_width = base_bit_width
        self.num_bit_options = num_bit_options
        self.coupling_strength = coupling_strength

        # Learnable layer importance scores for bit-width allocation
        self.importance_proj = nn.Sequential(
            nn.Linear(1, 16),
            nn.GELU(),
            nn.Linear(16, num_bit_options),
        )

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.zeros_(self.importance_proj[-1].bias)

    # ------------------------------------------------------------------
    # Bit-width allocation
    # ------------------------------------------------------------------

    def _allocate_bit_widths(
        self, weights: torch.Tensor
    ) -> torch.Tensor:
        """Allocate per-layer bit widths via C6 cyclic coupling.

        Parameters
        ----------
        weights : Tensor, shape ``(B, ...)`` — flattened per-layer weights.

        Returns
        -------
        Tensor of shape ``(B,)`` with integer bit widths.
        """
        # Compute importance: Frobenius norm per batch element
        w_flat = weights.view(weights.shape[0], -1)
        importance = w_flat.norm(dim=-1, keepdim=True)  # (B, 1)

        # Project to bit-width logits
        logits = self.importance_proj(importance)  # (B, num_bit_options)
        idx = logits.argmax(dim=-1)  # (B,)

        # Map indices through C6 cyclic permutation
        c6_perm = torch.arange(self.num_bit_options, device=weights.device)
        c6_perm = (c6_perm + int(CYCLIC_CONSTANT_142857 % self.num_bit_options)) % self.num_bit_options
        idx = c6_perm[idx]

        # Look up actual bit widths
        bits = torch.tensor(
            self.BIT_OPTIONS[:self.num_bit_options],
            device=weights.device,
            dtype=torch.float32,
        )
        return bits[idx.long()]

    # ------------------------------------------------------------------
    # Core quantise / dequantise
    # ------------------------------------------------------------------

    @staticmethod
    def _symmetric_quantize(
        w: torch.Tensor, bits: torch.Tensor
    ) -> torch.Tensor:
        """Per-element symmetric linear quantisation.

        Parameters
        ----------
        w : Tensor of any shape.
        bits : Tensor of shape ``(B,)`` — bit widths per batch element.

        Returns
        -------
        Quantised tensor with same shape as ``w``.
        """
        B = w.shape[0]
        levels = (2.0 ** bits - 1).clamp(min=1.0)  # (B,)

        # Per-batch max for scaling
        w_flat = w.view(B, -1)
        abs_max = w_flat.abs().max(dim=-1).values + 1e-12  # (B,)
        scale = abs_max / levels  # (B,)

        # Quantise
        q = torch.round(w_flat / scale.unsqueeze(-1))
        q = q * scale.unsqueeze(-1)

        return q.view_as(w)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, weights: torch.Tensor) -> QuantResult:
        """Quantise and dequantise weights with C6 bit allocation.

        Parameters
        ----------
        weights : Tensor, shape ``(B, ...)`` where B is the batch/layer dim.

        Returns
        -------
        QuantResult with compressed, recovered, bit_widths, metrics.
        """
        original_shape = weights.shape
        B = weights.shape[0]

        # Allocate bit widths
        bit_widths = self._allocate_bit_widths(weights)  # (B,)

        # Symmetric quantise (detached for "compressed" output)
        compressed = self._symmetric_quantize(
            weights.detach(), bit_widths
        )

        # Recovered = re-quantise original (gradient flows here)
        recovered = self._symmetric_quantize(weights, bit_widths)

        # Metrics
        w_flat = weights.view(B, -1)
        c_flat = compressed.view(B, -1)
        error = (w_flat - c_flat).norm(dim=-1)
        fidelity = 1.0 - error / (w_flat.norm(dim=-1) + 1e-12)
        compression_ratio = 32.0 / bit_widths.float().mean()

        metrics = {
            "mean_fidelity": fidelity.mean().item(),
            "mean_bit_width": bit_widths.float().mean().item(),
            "compression_ratio": compression_ratio.item(),
        }

        return QuantResult(
            compressed=compressed,
            recovered=recovered,
            bit_widths=bit_widths,
            metrics=metrics,
        )

    # ------------------------------------------------------------------
    # Fake quantize (straight-through estimator)
    # ------------------------------------------------------------------

    def fake_quantize(self, weights: torch.Tensor) -> torch.Tensor:
        """Training-time fake quantisation (STE).

        Forward pass: quantise & dequantise (simulates inference).
        Backward pass: gradient flows straight through to ``weights``.

        Parameters
        ----------
        weights : Tensor, shape ``(B, ...)``.

        Returns
        -------
        Tensor same shape as ``weights``, differentiable.
        """
        bit_widths = self._allocate_bit_widths(weights)
        compressed = self._symmetric_quantize(weights.detach(), bit_widths)
        # Straight-through estimator
        return weights + (compressed - weights).detach()

    def extra_repr(self) -> str:
        return (
            f"base_bit_width={self.base_bit_width}, "
            f"num_bit_options={self.num_bit_options}"
        )
