"""
TaiChi Router Module — PyTorch nn.Module wrapper.

Wraps the three-mode MoE routing logic of taichi-router into a
differentiable PyTorch module.  Input features are classified into
**Steady**, **Transitional**, or **Perturbation** modes, and a
soft coupling strength is produced for batched inputs.

Mathematical basis
------------------
For each batch element the router computes an energy scalar:

    E = ‖x[:6]‖₂²

This is mapped to one of three modes via thresholds θ_s and θ_p.
The coupling strength is derived from the softmax over a 3-class
energy embedding, with an entropy-regularisation term drawn from
C6 golden-ratio compensation (0.0618).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from taichi_matrix.constants import (
    GOLDEN_RATIO_COMPENSATION,
    PERTURBATION_THRESHOLD,
    STEADY_THRESHOLD,
    C6_ORDER,
)


@dataclass
class RoutingResult:
    """Output of :class:`TaiChiRouterModule`."""

    #: Mode index per batch element (0=Steady, 1=Transitional, 2=Perturbation).
    mode: torch.Tensor
    #: Per-element coupling strength, shape ``(B,)``.
    coupling_strength: torch.Tensor
    #: Soft routing weights, shape ``(B, 3)``.
    weights: torch.Tensor
    #: Per-expert output tensors (3 tensors of shape ``(B, D)``).
    expert_outputs: List[torch.Tensor]


class TaiChiRouterModule(nn.Module):
    """C6 MoE Router as a PyTorch module.

    Parameters
    ----------
    hidden_dim : int
        Feature dimension (input must have at least 6 channels).
    num_experts : int
        Number of routing experts (default 3, matching C6/2 modes).
    entropy_comp : float
        Entropy compensation applied to softmax weights.
    """

    MODE_NAMES = ("steady", "transitional", "perturbation")

    def __init__(
        self,
        hidden_dim: int = 64,
        num_experts: int = 3,
        entropy_comp: float = GOLDEN_RATIO_COMPENSATION,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.entropy_comp = entropy_comp

        # Energy projection: 6 → num_experts
        self.energy_proj = nn.Linear(6, num_experts, bias=False)
        # Per-expert shallow transform for demonstration
        self.expert_fcs = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(num_experts)]
        )

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """Xavier-initialise linear layers."""
        for p in self.parameters():
            if p.dim() >= 2:
                nn.init.xavier_uniform_(p)

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def compute_coupling_strength(self, features: torch.Tensor) -> torch.Tensor:
        """Compute per-batch coupling strength from first 6 features.

        Parameters
        ----------
        features : Tensor, shape ``(B, D)`` where D ≥ 6.

        Returns
        -------
        Tensor, shape ``(B,)`` — scalar coupling strength per element.
        """
        x6 = features[..., :6]
        logits = self.energy_proj(x6)          # (B, num_experts)
        logits = logits - self.entropy_comp * torch.log(
            logits.softmax(dim=-1) + 1e-8
        )
        weights = F.softmax(logits, dim=-1)    # (B, num_experts)
        # Coupling = weighted sum of normalised energy levels
        strength = (weights * torch.arange(
            self.num_experts, device=weights.device, dtype=weights.dtype
        ).unsqueeze(0)).sum(dim=-1)
        return strength / max(self.num_experts - 1, 1)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> RoutingResult:
        """Route input through C6 MoE experts.

        Parameters
        ----------
        x : Tensor, shape ``(B, D)`` or ``(D,)``.
            Input features.
        mask : Tensor, optional, shape ``(B,)``.
            Binary mask; masked positions default to Steady.

        Returns
        -------
        RoutingResult with mode, coupling_strength, weights, expert_outputs.
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)

        B, D = x.shape

        # 1. Routing weights
        x6 = x[..., :6]
        logits = self.energy_proj(x6)
        logits = logits - self.entropy_comp * torch.log(
            logits.softmax(dim=-1) + 1e-8
        )
        weights = F.softmax(logits, dim=-1)  # (B, num_experts)

        # 2. Mode classification from energy
        energy = (x6 ** 2).sum(dim=-1)  # (B,)
        mode = torch.zeros(B, dtype=torch.long, device=x.device)
        mode[energy >= STEADY_THRESHOLD] = 1
        mode[energy >= PERTURBATION_THRESHOLD] = 2
        if mask is not None:
            mode = mode * mask.long()

        # 3. Coupling strength
        coupling = self.compute_coupling_strength(x)

        # 4. Expert forward
        expert_outputs: List[torch.Tensor] = []
        for i, fc in enumerate(self.expert_fcs):
            # Gate each expert by its weight (grad-friendly)
            out = fc(x) * weights[:, i].unsqueeze(-1)
            expert_outputs.append(out)

        return RoutingResult(
            mode=mode,
            coupling_strength=coupling,
            weights=weights,
            expert_outputs=expert_outputs,
        )

    def extra_repr(self) -> str:
        return (
            f"hidden_dim={self.hidden_dim}, num_experts={self.num_experts}, "
            f"entropy_comp={self.entropy_comp:.4f}"
        )
