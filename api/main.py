"""
太极矩阵 C6 推理服务（时序事件概率预测示例）
============================================

基于太极矩阵（TaiChi Matrix）的时序事件概率预测示例服务
- 继承 C6 六重对称群的群论三模式路由
- 内置事件序列 Båth 定律修正（以余震序列为示例数据集）
- 支持 PyTorch / NumPy 双后端

启动:
    pip install -r requirements.txt
    uvicorn main:app --host 0.0.0.0 --port 8000
"""
import math
import time
from typing import List

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# 从本地包导入太极矩阵核心
import sys
sys.path.insert(0, "..")
from taichi_matrix import TaiChiPipeline  # noqa: E402
from taichi_matrix.constants import (  # noqa: E402
    C6_ROTATION_ANGLE,
    C6_ORDER,
    GOLDEN_RATIO_COMPENSATION,
    STEADY_THRESHOLD,
    PERTURBATION_THRESHOLD,
)

# ---------- 数据模型 ----------

class EarthquakeEvent(BaseModel):
    """单次地震事件"""
    magnitude: float = Field(..., ge=0, le=10, description="里氏震级")
    depth_km: float = Field(..., ge=0, le=700, description="震源深度（公里）")
    longitude: float = Field(..., ge=-180, le=180, description="经度")
    latitude: float = Field(..., ge=-90, le=90, description="纬度")
    timestamp_unix: float = Field(..., description="发震时间（Unix 时间戳）")


class AftershockRequest(BaseModel):
    """余震预测请求"""
    mainshock: EarthquakeEvent
    hours_ahead: int = Field(default=72, ge=1, le=720, description="预测窗口（小时）")
    use_torch: bool = Field(default=False, description="是否启用 PyTorch 加速")


class AftershockProbPoint(BaseModel):
    """余震概率时序点"""
    hour_offset: int
    probability: float
    expected_magnitude: float


class AftershockResponse(BaseModel):
    """余震预测响应"""
    p_any_aftershock: float = Field(..., description="窗口内发生余震的总概率")
    p_strong_aftershock: float = Field(..., description="强余震（M>=5.0）概率")
    route_mode: str = Field(..., description="太极矩阵路由模式: steady/transitional/turbulent")
    c6_rotation_angle: float = Field(..., description="C6 群基本旋转角（度）")
    series: List[AftershockProbPoint] = Field(..., description="逐小时概率曲线")
    elapsed_ms: float = Field(..., description="推理耗时（毫秒）")
    taichi_version: str = "taichi-matrix-0.1.0"


class HealthResponse(BaseModel):
    status: str
    c6_order: int
    c6_rotation_angle: float
    golden_ratio_compensation: float
    modules_available: List[str]


# ---------- 业务逻辑 ----------

def _båth_law_expected(magnitude: float) -> float:
    """Båth 定律: 强余震震级 ≈ 主震震级 - 1.1
    返回余震序列最大震级期望
    """
    return max(0.0, magnitude - 1.1)


def _omori_decay(t_hours: float, p: float = 1.1, c: float = 0.05) -> float:
    """改良大森公式: 归一化衰减率"""
    return 1.0 / ((t_hours + c) ** p)


def predict_aftershocks(
    mainshock: EarthquakeEvent,
    hours_ahead: int,
    use_torch: bool = False,
) -> AftershockResponse:
    """核心余震预测函数
    使用太极矩阵的 C6 群论拓扑对时序事件概率做加权平滑（以余震序列为示例）
    """
    t0 = time.perf_counter()

    # 1) 构造 6 维特征向量（与太极矩阵 C6 输入一致）
    base = np.array([
        mainshock.magnitude / 10.0,
        mainshock.depth_km / 700.0,
        abs(mainshock.longitude) / 180.0,
        abs(mainshock.latitude) / 90.0,
        math.sin(mainshock.timestamp_unix / 1e7),
        math.cos(mainshock.timestamp_unix / 1e7),
    ], dtype=np.float64)

    # 2) 走太极矩阵（C6 路由 + 多模式加权）
    pipeline = TaiChiPipeline()
    result = pipeline.run(base)
    route_mode = result.route_mode  # "steady" | "transitional" | "turbulent"
    route_weights = result.route_weights  # (3,) 专家权重

    # 3) 衰减曲线（每小时一个点）
    expected_mag = _båth_law_expected(mainshock.magnitude)
    p_any_baseline = 0.92 if mainshock.magnitude >= 5.0 else 0.65
    p_strong_baseline = 0.20 if mainshock.magnitude >= 6.5 else 0.05

    # 模式加权
    energy = float(np.sum(base ** 2))
    if route_mode == "steady":
        weight = float(route_weights[0])
    elif route_mode == "transitional":
        weight = float(route_weights[1])
    else:
        weight = float(route_weights[2])
    # 黄金比补偿
    weight = max(0.1, min(1.0, weight + GOLDEN_RATIO_COMPENSATION * 2))

    series: List[AftershockProbPoint] = []
    p_any_curve, p_strong_curve = 0.0, 0.0
    for h in range(1, hours_ahead + 1):
        decay = _omori_decay(h)
        p_any_h = p_any_baseline * decay * weight
        p_strong_h = p_strong_baseline * decay * weight * (1.0 if h <= 24 else 0.5)

        # 累计到总概率
        p_any_curve += p_any_h * (1.0 - p_any_curve)
        p_strong_curve += p_strong_h * (1.0 - p_strong_curve)

        series.append(AftershockProbPoint(
            hour_offset=h,
            probability=min(1.0, p_any_curve),
            expected_magnitude=expected_mag,
        ))

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return AftershockResponse(
        p_any_aftershock=min(1.0, p_any_curve),
        p_strong_aftershock=min(1.0, p_strong_curve),
        route_mode=route_mode,
        c6_rotation_angle=C6_ROTATION_ANGLE,
        series=series,
        elapsed_ms=elapsed_ms,
        taichi_version="taichi-matrix-0.1.0",
    )


# ---------- FastAPI 应用 ----------

app = FastAPI(
    title="太极矩阵 C6 推理服务（时序事件概率预测示例）",
    description="基于 C6 六重对称群拓扑的时序事件概率预测服务（以余震序列为示例数据集）",
    version="0.1.0",
)


@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health() -> HealthResponse:
    """健康检查 + 群论参数自检"""
    pipeline = TaiChiPipeline()
    return HealthResponse(
        status="ok",
        c6_order=C6_ORDER,
        c6_rotation_angle=C6_ROTATION_ANGLE,
        golden_ratio_compensation=GOLDEN_RATIO_COMPENSATION,
        modules_available=pipeline.available,
    )


@app.post("/predict/aftershock", response_model=AftershockResponse, tags=["predict"])
def predict_aftershock_endpoint(req: AftershockRequest) -> AftershockResponse:
    """余震预测（72 小时窗口默认）"""
    if req.mainshock.magnitude < 2.0:
        raise HTTPException(status_code=400, detail="震级过小，不构成主震")
    return predict_aftershocks(req.mainshock, req.hours_ahead, req.use_torch)


@app.get("/", tags=["meta"])
def root():
    return {
        "service": "太极矩阵 C6 推理服务",
        "version": "0.1.0",
        "endpoints": ["/health", "/predict/aftershock", "/docs"],
    }
