"""
TaiChi Torch Adapter — PyTorch wrappers for the 5 TaiChi Matrix modules.

This sub-package re-exports nn.Module adapters and a unified pipeline
so that TaiChi Matrix can be dropped into any PyTorch training / inference
graph with end-to-end differentiability.

**Lazy import guard**: ``torch >= 2.0.0`` must be installed; otherwise
importing this package raises ``ImportError`` immediately.
"""

from __future__ import annotations

__all__ = [
    "TaiChiRouterModule",
    "TaiChiMTPModule",
    "TaiChiQuantModule",
    "HexAttentionModule",
    "TaiChiCorrectModule",
    "TaiChiTorchPipeline",
]

# ---------------------------------------------------------------------------
# Soft dependency guard — torch is an optional extra.
# ---------------------------------------------------------------------------

def __getattr__(name: str):
    """Lazy-load modules so the top-level package can be imported even
    without torch installed ( AttributeError is raised only on access )."""
    if name in __all__:
        try:
            import torch  # noqa: F811  — first import triggers real work
        except ImportError as exc:
            raise ImportError(
                "PyTorch >= 2.0 is required to use taichi_matrix.torch_adapter.  "
                "Install it with:  pip install 'taichi-matrix[torch]'"
            ) from exc

        # First access → populate module globals so subsequent lookups are fast.
        import importlib
        from taichi_matrix.torch_adapter.router_module import TaiChiRouterModule
        from taichi_matrix.torch_adapter.mtp_module import TaiChiMTPModule
        from taichi_matrix.torch_adapter.quant_module import TaiChiQuantModule
        from taichi_matrix.torch_adapter.hex_attention_module import HexAttentionModule
        from taichi_matrix.torch_adapter.correct_module import TaiChiCorrectModule
        from taichi_matrix.torch_adapter.pipeline import TaiChiTorchPipeline

        _mapping = {
            "TaiChiRouterModule": TaiChiRouterModule,
            "TaiChiMTPModule": TaiChiMTPModule,
            "TaiChiQuantModule": TaiChiQuantModule,
            "HexAttentionModule": HexAttentionModule,
            "TaiChiCorrectModule": TaiChiCorrectModule,
            "TaiChiTorchPipeline": TaiChiTorchPipeline,
        }
        import sys
        mod = sys.modules[__name__]
        for k, v in _mapping.items():
            setattr(mod, k, v)

        return _mapping[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# When torch IS available, make `from taichi_matrix.torch_adapter import X`
# work eagerly by loading everything at import time.
def _eager_load():
    import torch  # noqa: F401
    _ = [__getattr__(n) for n in __all__]

try:
    _eager_load()
except Exception:
    pass  # torch not installed — access will trigger __getattr__
finally:
    del _eager_load
