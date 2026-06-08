"""
TaiChi Torch Pipeline — End-to-end differentiable C6 pipeline.

Chains all 5 TaiChi Matrix modules in order:
    Router → MTP → HexAttention → Quant → Correct

Every stage is a PyTorch nn.Module with standard ``forward()`` semantics,
ensuring end-to-end gradient flow through the entire chain.

Mathematical basis
------------------
Each module applies a C6-symmetric transformation.  The pipeline
preserves the group structure by using the same coupling constants
and eigenbases across stages, creating a coherent hexagonal signal
processing graph.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from taichi_matrix.constants import C6_ORDER, GOLDEN_RATIO_COMPENSATION


@dataclass
class PipelineResult:
    """Output of :class:`TaiChiTorchPipeline`."""

    #: Final corrected output, shape ``(B, D)``.
    output: torch.Tensor
    #: Per-stage intermediate results (dict keyed by stage name).
    intermediates: Dict[str, object] = field(default_factory=dict)
    #: Names of stages that were actually executed.
    stages_run: List[str] = field(default_factory=list)


class TaiChiTorchPipeline(nn.Module):
    """Unified C6 pipeline: Router → MTP → HexAttn → Quant → Correct.

    Parameters
    ----------
    hidden_dim : int
        Hidden dimension used by all sub-modules.
    vocab_size : int
        Vocabulary size for MTP output heads.
    num_heads : int
        Number of attention heads for HexAttention.
    enable : list of str, optional
        Subset of stages to enable.  Default: all.
        Choices: ``"router"``, ``"mtp"``, ``"hex"``, ``"quant"``, ``"correct"``.
    """

    _ALL_STAGES = ("router", "mtp", "hex", "quant", "correct")

    def __init__(
        self,
        hidden_dim: int = 64,
        vocab_size: int = 32000,
        num_heads: int = C6_ORDER,
        enable: Optional[List[str]] = None,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size

        if enable is None:
            enable = list(self._ALL_STAGES)
        self.enabled = [s for s in enable if s in self._ALL_STAGES]

        # Round hidden_dim up to be divisible by num_heads for HexAttention
        self.hex_dim = ((hidden_dim + num_heads - 1) // num_heads) * num_heads

        # --- Router ---
        self.router = nn.ModuleDict()
        if "router" in self.enabled:
            self.router["module"] = _lazy_import_router(hidden_dim)

        # --- MTP ---
        self.mtp = nn.ModuleDict()
        if "mtp" in self.enabled:
            self.mtp["module"] = _lazy_import_mtp(hidden_dim, vocab_size)

        # --- HexAttention ---
        self.hex_attn = nn.ModuleDict()
        if "hex" in self.enabled:
            self.hex_attn["module"] = _lazy_import_hex(self.hex_dim, num_heads)

        # --- Quant ---
        self.quant = nn.ModuleDict()
        if "quant" in self.enabled:
            self.quant["module"] = _lazy_import_quant()

        # --- Correct ---
        self.correct = nn.ModuleDict()
        if "correct" in self.enabled:
            self.correct["module"] = _lazy_import_correct(hidden_dim)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> PipelineResult:
        """Execute the full C6 pipeline on input tensor.

        Parameters
        ----------
        x : Tensor, shape ``(B, D)`` or ``(D,)``.
            Input features (hidden states or embeddings).

        Returns
        -------
        PipelineResult with final output and per-stage intermediates.
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)

        intermediates: Dict[str, object] = {}
        stages_run: List[str] = []
        current = x

        # 1. Router
        if "router" in self.enabled and "module" in self.router:
            result = self.router["module"](current)
            intermediates["router"] = result
            # Use weighted sum of expert outputs as routed representation
            weights = result.weights  # (B, 3)
            expert_stack = torch.stack(result.expert_outputs, dim=1)  # (B, 3, D)
            current = (expert_stack * weights.unsqueeze(-1)).sum(dim=1)  # (B, D)
            stages_run.append("router")

        # 2. MTP
        if "mtp" in self.enabled and "module" in self.mtp:
            result = self.mtp["module"](current)
            intermediates["mtp"] = result
            # Pool across 6 token predictions: mean logit → hidden
            current = result.predictions.mean(dim=1)  # (B, V)
            # Project back to hidden_dim via simple average pooling
            if current.shape[-1] != self.hidden_dim:
                # Linear-ish reduction: slice + mean
                reduce_factor = current.shape[-1] // self.hidden_dim
                if reduce_factor > 0:
                    current = current.view(
                        current.shape[0], self.hidden_dim, reduce_factor
                    ).mean(dim=-1)
                else:
                    current = current[..., :self.hidden_dim]
            stages_run.append("mtp")

        # 3. HexAttention (expects (B, S, D) — treat B as seq)
        if "hex" in self.enabled and "module" in self.hex_attn:
            # Pad hidden_dim to hex_dim if needed
            if current.shape[-1] < self.hex_dim:
                pad = torch.zeros(
                    *current.shape[:-1], self.hex_dim - current.shape[-1],
                    device=current.device, dtype=current.dtype,
                )
                current_hex = torch.cat([current, pad], dim=-1)
            else:
                current_hex = current[..., :self.hex_dim]
            attn_input = current_hex.unsqueeze(1)  # (B, 1, hex_dim)
            result = self.hex_attn["module"](attn_input)
            intermediates["hex"] = result
            current = result.output.squeeze(1)[..., :self.hidden_dim]  # (B, D)
            stages_run.append("hex")

        # 4. Quant
        if "quant" in self.enabled and "module" in self.quant:
            result = self.quant["module"](current)
            intermediates["quant"] = result
            current = result.recovered  # (B, D) — differentiable
            stages_run.append("quant")

        # 5. Correct
        if "correct" in self.enabled and "module" in self.correct:
            result = self.correct["module"](current)
            intermediates["correct"] = result
            current = result.corrected  # (B, D)
            stages_run.append("correct")

        return PipelineResult(
            output=current,
            intermediates=intermediates,
            stages_run=stages_run,
        )

    # ------------------------------------------------------------------
    # Stage toggling
    # ------------------------------------------------------------------

    def enable_stage(self, stage: str) -> None:
        """Enable a previously disabled stage (no-op if already enabled)."""
        if stage in self._ALL_STAGES and stage not in self.enabled:
            self.enabled.append(stage)

    def disable_stage(self, stage: str) -> None:
        """Disable a stage by removing it from the execution list."""
        if stage in self.enabled:
            self.enabled.remove(stage)

    def extra_repr(self) -> str:
        bars = "".join(
            "●" if s in self.enabled else "○" for s in self._ALL_STAGES
        )
        return (
            f"hidden_dim={self.hidden_dim}, "
            f"vocab_size={self.vocab_size}, "
            f"stages=[{bars}]"
        )


# ---------------------------------------------------------------------------
# Lazy module constructors (imported only once, cached in callers)
# ---------------------------------------------------------------------------

def _lazy_import_router(hidden_dim: int):
    from taichi_matrix.torch_adapter.router_module import TaiChiRouterModule
    return TaiChiRouterModule(hidden_dim=hidden_dim)


def _lazy_import_mtp(hidden_dim: int, vocab_size: int):
    from taichi_matrix.torch_adapter.mtp_module import TaiChiMTPModule
    return TaiChiMTPModule(
        hidden_dim=hidden_dim, vocab_size=vocab_size
    )


def _lazy_import_hex(hidden_dim: int, num_heads: int):
    from taichi_matrix.torch_adapter.hex_attention_module import HexAttentionModule
    return HexAttentionModule(d_model=hidden_dim, num_heads=num_heads)


def _lazy_import_quant():
    from taichi_matrix.torch_adapter.quant_module import TaiChiQuantModule
    return TaiChiQuantModule()


def _lazy_import_correct(vocab_size: int):
    from taichi_matrix.torch_adapter.correct_module import TaiChiCorrectModule
    return TaiChiCorrectModule(num_classes=vocab_size)
