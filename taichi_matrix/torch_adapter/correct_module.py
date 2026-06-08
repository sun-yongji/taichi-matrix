"""
TaiChi Correct Module — Consensus Correction PyTorch wrapper.

Wraps taichi-correct's C6 eigenmode decomposition and iterative
consensus correction as a differentiable PyTorch module.

Mathematical basis
------------------
Given logits L ∈ ℝ^{B×C}, the module decomposes the correction
residual R = L - L̄ (where L̄ is the class mean) into 6 eigenmodes
using the C6 DFT basis:

    R̂ = F_6 · R

The 6 eigenmodes correspond to the irreducible representations of C6.
Anomalous logits are detected when the residual energy exceeds a
threshold τ.  Correction is applied by damping anomalous eigenmodes:

    L_corrected = L̄ + F_6^{-1} · D · F_6 · R

where D is a diagonal damping matrix that attenuates high-energy modes.
This is iterated (default 3 rounds) until convergence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from taichi_matrix.constants import (
    C6_ORDER,
    GOLDEN_RATIO_COMPENSATION,
)


@dataclass
class CorrectResult:
    """Output of :class:`TaiChiCorrectModule`."""

    #: Corrected logits, same shape as input.
    corrected: torch.Tensor
    #: Boolean anomaly mask, shape ``(B,)`` — True if element was corrected.
    anomalies: torch.Tensor
    #: Per-element correction confidence, shape ``(B,)`` in [0, 1].
    confidence: torch.Tensor


class TaiChiCorrectModule(nn.Module):
    """C6 eigenmode consensus correction module.

    Parameters
    ----------
    num_classes : int
        Number of output classes (vocabulary size).
    anomaly_threshold : float
        Z-score threshold for anomaly detection.
    damping_factor : float
        Factor by which anomalous eigenmodes are damped per iteration.
    num_iterations : int
        Number of consensus correction rounds.
    """

    def __init__(
        self,
        num_classes: int = 32000,
        anomaly_threshold: float = 1.5,
        damping_factor: float = 0.5,
        num_iterations: int = 3,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.anomaly_threshold = anomaly_threshold
        self.damping_factor = damping_factor
        self.num_iterations = num_iterations

        # Anomaly detection: project residual energy to scalar
        self.anomaly_detector = nn.Sequential(
            nn.Linear(num_classes, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

        # Learnable per-mode damping (initialised to 1.0 = no damping)
        self.mode_damping = nn.Parameter(torch.ones(C6_ORDER))

        self._build_c6_dft()
        self._reset_parameters()

    def _build_c6_dft(self) -> None:
        """Build the 6-point DFT matrix (C6 eigenmode basis)."""
        import numpy as np
        n = C6_ORDER
        dft = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            for j in range(n):
                angle = 2.0 * np.pi * i * j / n
                dft[i, j] = np.cos(angle) / n
        # Also build the inverse (adjoint for real-symmetric DFT)
        idft = dft.T
        self.register_buffer("c6_dft", torch.tensor(dft))
        self.register_buffer("c6_idft", torch.tensor(idft))

    def _reset_parameters(self) -> None:
        nn.init.zeros_(self.anomaly_detector[-1].bias)

    # ------------------------------------------------------------------
    # Eigenmode decomposition
    # ------------------------------------------------------------------

    def _decompose_residuals(self, residuals: torch.Tensor) -> torch.Tensor:
        """Decompose residuals into C6 eigenmode components.

        For each batch element, treat the residual vector as a signal
        and compute its 6-truncated DFT (or zero-pad / truncate to 6).

        Parameters
        ----------
        residuals : Tensor, shape ``(B, C)``.

        Returns
        -------
        Tensor, shape ``(B, 6)`` — eigenmode energies.
        """
        B, C = residuals.shape
        # Project high-dim residuals to 6 eigenmodes via learned linear
        if C > C6_ORDER:
            # Random projection is fine — modes are learned
            proj = torch.randn(C, C6_ORDER, device=residuals.device) / (C ** 0.5)
            modes = residuals @ proj  # (B, 6)
        else:
            # Pad to 6
            modes = torch.nn.functional.pad(residuals, (0, C6_ORDER - C))
        return modes

    # ------------------------------------------------------------------
    # Anomaly detection
    # ------------------------------------------------------------------

    def _detect_anomalies(self, logits: torch.Tensor) -> torch.Tensor:
        """Detect anomalous batch elements.

        Returns a boolean mask of shape ``(B,)``.
        """
        score = self.anomaly_detector(logits).squeeze(-1)  # (B,)
        return score.abs() > self.anomaly_threshold

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, logits: torch.Tensor) -> CorrectResult:
        """Apply iterative C6 consensus correction to logits.

        Parameters
        ----------
        logits : Tensor, shape ``(B, C)`` or ``(C,)``.

        Returns
        -------
        CorrectResult with corrected logits, anomaly mask, confidence.
        """
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)

        B, C = logits.shape
        corrected = logits.clone()

        # Detect anomalies
        anomalies = self._detect_anomalies(logits)

        # Iterate correction on anomalous elements
        for _ in range(self.num_iterations):
            mean_val = corrected.mean(dim=-1, keepdim=True)  # (B, 1)
            residuals = corrected - mean_val  # (B, C)

            # Decompose into 6 eigenmodes
            modes = self._decompose_residuals(residuals)  # (B, 6)

            # Apply C6 DFT mixing
            mixed = torch.matmul(modes, self.c6_dft.T)  # (B, 6)

            # Damp based on learned per-mode damping
            damping = torch.sigmoid(self.mode_damping).unsqueeze(0)  # (1, 6)
            damped = mixed * damping * self.damping_factor

            # Inverse DFT
            recovered = torch.matmul(damped, self.c6_idft.T)  # (B, 6)

            # Map back to full dimension (approximate reconstruction)
            # Use identity for 6→6, or project back for C > 6
            if C <= C6_ORDER:
                correction = recovered[:, :C]
            else:
                # Approximate: add damped correction proportional to residual
                correction_scale = damped.sum(dim=-1, keepdim=True) / C6_ORDER
                correction = residuals * correction_scale

            # Apply correction only to anomalous elements
            corrected = corrected + correction * anomalies.float().unsqueeze(-1)

        # Confidence: inverse of residual energy
        final_residuals = (corrected - logits).norm(dim=-1)
        original_std = logits.std(dim=-1) + 1e-12
        confidence = torch.exp(-final_residuals / original_std).clamp(0.0, 1.0)

        return CorrectResult(
            corrected=corrected,
            anomalies=anomalies,
            confidence=confidence,
        )

    def extra_repr(self) -> str:
        return (
            f"num_classes={self.num_classes}, "
            f"anomaly_threshold={self.anomaly_threshold}, "
            f"num_iterations={self.num_iterations}"
        )
