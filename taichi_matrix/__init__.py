"""
TaiChi Matrix — Eastern Mathematics-driven AI Infrastructure Toolkit.

Supports Ascend NPU, CUDA, and CPU via auto device detection.

Orchestrates 5 C6-symmetric modules:
  M1 Router  → MoE dynamic routing with entropy balance
  M2 MTP     → Multi-token prediction with hex-depthed coupling
  M3 Quant   → Hex-yao quantization with C6 coupling groups
  M4 HexAttn → Hexagonal diagonal attention with HexRoPE
  M5 Correct → C6 error correction with eigenmode decomposition

CCF OSS 2026 · Apache 2.0
"""

from taichi_matrix.pipeline import TaiChiPipeline, PipelineResult
from taichi_matrix import device_utils

__all__ = ["TaiChiPipeline", "PipelineResult", "device_utils"]
__version__ = "0.1.0"
