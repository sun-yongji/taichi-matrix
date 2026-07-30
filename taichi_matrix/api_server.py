"""
TaiChi Matrix — FastAPI Inference Server.

Provides a REST API for running the TaiChi pipeline on Ascend / CUDA / CPU.

Start:
    python -m taichi_matrix.api_server --host 0.0.0.0 --port 8000

Request:
    curl -X POST http://localhost:8000/pipeline \\
        -H "Content-Type: application/json" \\
        -d '{"input": [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]]}'
"""

from __future__ import annotations

import argparse
import time
from typing import List, Optional

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from taichi_matrix.device_utils import device_info, get_device
from taichi_matrix.pipeline import TaiChiPipeline

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="TaiChi Matrix API",
    description="C6 symmetry-driven AI infrastructure optimization pipeline",
    version="0.1.0",
)

_pipeline: Optional[TaiChiPipeline] = None


def get_pipeline() -> TaiChiPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = TaiChiPipeline()
    return _pipeline


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class PipelineRequest(BaseModel):
    input: List[List[float]] = Field(
        ...,
        description="Input tensor as a 2D list of floats",
        example=[[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]],
    )
    enable: Optional[List[str]] = Field(
        None,
        description="Stages to enable: router, mtp, hex, quant, correct",
    )


class TimingInfo(BaseModel):
    router: float = 0.0
    mtp: float = 0.0
    hex: float = 0.0
    quant: float = 0.0
    correct: float = 0.0
    total_ms: float = 0.0


class PipelineResponse(BaseModel):
    route_mode: str
    route_weights: List[float]
    attention_delta: float
    confidence: float
    residue_reduction: float
    modules_available: List[str]
    timings: TimingInfo
    device: str


class DeviceInfoResponse(BaseModel):
    device: str
    ascend_available: bool
    cuda_available: bool
    cuda_device_count: int
    npu_device_count: Optional[int] = None
    npu_device_name: Optional[str] = None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/", tags=["info"])
async def root():
    return {
        "service": "TaiChi Matrix",
        "version": "0.1.0",
        "docs": "/docs",
        "device": str(get_device()),
    }


@app.get("/device", response_model=DeviceInfoResponse, tags=["info"])
async def get_device_info():
    """Return device detection info."""
    return DeviceInfoResponse(**device_info())


@app.post("/pipeline", response_model=PipelineResponse, tags=["inference"])
async def run_pipeline(req: PipelineRequest):
    """Run the TaiChi inference pipeline."""
    try:
        x = np.array(req.input, dtype=np.float64)
        if x.ndim not in (1, 2):
            raise HTTPException(
                status_code=400,
                detail=f"Input must be 1D or 2D, got shape {x.shape}",
            )

        t0 = time.perf_counter()
        pipeline = get_pipeline()
        result = pipeline.run(x)
        total_ms = (time.perf_counter() - t0) * 1000

        return PipelineResponse(
            route_mode=result.route_mode,
            route_weights=result.route_weights.tolist(),
            attention_delta=result.attention_delta,
            confidence=result.confidence,
            residue_reduction=result.residue_reduction,
            modules_available=result.modules_available,
            timings=TimingInfo(
                **result.timings,
                total_ms=round(total_ms, 3),
            ),
            device=str(get_device()),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="TaiChi Matrix API Server")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address")
    parser.add_argument("--port", type=int, default=8000, help="Port")
    parser.add_argument("--reload", action="store_true", help="Hot reload")
    args = parser.parse_args()

    print(f"🚀 TaiChi Matrix API starting on http://{args.host}:{args.port}")
    print(f"   Device: {get_device()}")
    uvicorn.run(
        "taichi_matrix.api_server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
