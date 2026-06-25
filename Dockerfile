# 太极矩阵地震余震预测 API Dockerfile
# 多阶段构建以减小最终镜像体积
FROM python:3.11-slim AS builder

WORKDIR /app

# 先装构建依赖与 pip
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc git && \
    rm -rf /var/lib/apt/lists/*

COPY api/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip wheel --no-cache-dir --wheel-dir /wheels -r /app/requirements.txt

# -------- 运行时镜像 --------
FROM python:3.11-slim

WORKDIR /app

# 把本地太极矩阵包和 api 代码都带进来
COPY taichi_matrix/ /app/taichi_matrix/
COPY api/ /app/api/

# 从构建阶段复制 wheels 并安装
COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir --no-index --find-links=/wheels \
    fastapi==0.115.0 uvicorn[standard]==0.30.6 pydantic==2.9.2 numpy && \
    pip install --no-cache-dir -e /app/taichi_matrix && \
    rm -rf /wheels

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    API_PORT=8000

EXPOSE 8000

# 健康检查
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health').read()" \
    || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
