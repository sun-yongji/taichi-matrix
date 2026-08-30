"""
TaiChi Matrix — Device Utilities for Ascend NPU / CUDA / CPU.

Auto-detection order:
    1. Ascend NPU (torch_npu)
    2. NVIDIA CUDA
    3. CPU (fallback)

Usage:
    from taichi_matrix.device_utils import get_device, to_device

    device = get_device()
    tensor = torch.randn(32, 128, device=device)
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import torch

logger = logging.getLogger("taichi_matrix.device")

# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

_ASCEND_AVAILABLE: Optional[bool] = None
_DEVICE_CACHE: Optional[torch.device] = None


def _check_ascend() -> bool:
    """Return True if Ascend NPU (torch_npu) is available."""
    global _ASCEND_AVAILABLE
    if _ASCEND_AVAILABLE is not None:
        return _ASCEND_AVAILABLE
    try:
        import torch_npu  # noqa: F401

        if torch.npu.is_available():
            _ASCEND_AVAILABLE = True
            logger.info("Ascend NPU detected ✓")
            return True
    except (ImportError, RuntimeError, AttributeError):
        pass
    _ASCEND_AVAILABLE = False
    logger.info("Ascend NPU not available")
    return False


def get_device() -> torch.device:
    """Auto-select device: Ascend NPU → CUDA → CPU."""
    global _DEVICE_CACHE
    if _DEVICE_CACHE is not None:
        return _DEVICE_CACHE

    if _check_ascend():
        _DEVICE_CACHE = torch.device("npu:0")
    elif torch.cuda.is_available():
        _DEVICE_CACHE = torch.device("cuda:0")
        logger.info("CUDA detected ✓")
    else:
        _DEVICE_CACHE = torch.device("cpu")
        logger.info("Using CPU")

    return _DEVICE_CACHE


def to_device(
    obj: Any,
    device: Optional[torch.device] = None,
) -> Any:
    """Move a tensor or module to the target device.

    Parameters
    ----------
    obj : torch.Tensor | torch.nn.Module | Any
        Object to move.
    device : torch.device, optional
        Target device.  Auto-detected if None.

    Returns
    -------
    Object on the target device.
    """
    if device is None:
        device = get_device()
    if isinstance(obj, (torch.Tensor, torch.nn.Module)):
        return obj.to(device)
    return obj


# ---------------------------------------------------------------------------
# Benchmark helper
# ---------------------------------------------------------------------------

def device_info() -> dict:
    """Return a dict with device detection results."""
    info = {
        "device": str(get_device()),
        "ascend_available": _check_ascend(),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }
    if _check_ascend():
        try:
            import torch_npu  # noqa: F401
            info["npu_device_count"] = torch.npu.device_count()
            info["npu_device_name"] = torch.npu.get_device_name(0)
        except Exception:
            pass
    return info
