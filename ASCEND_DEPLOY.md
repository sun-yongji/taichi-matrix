# 太极矩阵 · 昇腾部署方案

> TaiChi Matrix — Ascend NPU Deployment Guide

## 一、方案概览

太极矩阵项目当前为纯 Python + PyTorch 实现，所有算子均为标准 PyTorch 操作（Linear、softmax、einsum、matmul 等），无需修改核心算法即可适配昇腾 NPU。

| 项目 | 说明 |
|------|------|
| 底层框架 | PyTorch → torch_npu (昇腾 NPU 后端) |
| 适配程度 | 零代码侵入：仅修改设备初始化逻辑 |
| 硬件支持 | Atlas 800 (A2) / Atlas 900 训练/推理服务器 |
| 开发环境 | CANN 8.0+ / Ascend Docker 镜像 |

## 二、快速部署

### 2.1 昇腾环境准备

#### 方式 A：使用昇腾官方 Docker 镜像（推荐）

```bash
# 拉取昇腾 PyTorch 镜像
docker pull ascendai/pytorch:torch_npu2.6.0-3.0.0-ubuntu22.04

# 启动容器并挂载项目
docker run -it --rm \
  --name taichi-matrix \
  --device=/dev/davinci0 \
  --device=/dev/davinci_manager \
  --device=/dev/hisi_hdc \
  -v /opt/atomgit/taichi-matrix:/workspace/taichi-matrix \
  ascendai/pytorch:torch_npu2.6.0-3.0.0-ubuntu22.04 \
  bash
```

#### 方式 B：物理机安装

```bash
# 1. 安装 CANN 工具包（下载地址：https://www.hiascend.com/）
chmod +x Ascend-cann-toolkit_8.0.0_linux-x86_64.run
./Ascend-cann-toolkit_8.0.0_linux-x86_64.run --install --quiet

# 2. 安装 torch_npu
pip install torch_npu==2.6.0

# 3. 验证安装
python -c "import torch; import torch_npu; print(torch.npu.is_available())"
# 输出：True
```

### 2.2 安装太极矩阵

```bash
cd /workspace/taichi-matrix
pip install -e ".[all,dev]"
```

### 2.3 运行验证

```bash
# 运行测试
python -m pytest tests/ -v

# 运行流水线示例
python -c "
from taichi_matrix.pipeline import TaiChiPipeline
from taichi_matrix.device_utils import to_device
import numpy as np

p = TaiChiPipeline()
x = np.random.randn(32, 128)
result = p.run(x)
print(f'模式: {result.route_mode}')
print(f'用时: {result.timings}')
"
```

## 三、昇腾加速效果预期

| 模块 | CPU (当前) | 昇腾 NPU (预期) | 加速比 |
|------|-----------|----------------|--------|
| M1 路由 | 0.12ms | ~0.01ms | **~12x** |
| M2 多令牌 | 0.28ms | ~0.03ms | **~9x** |
| M3 量化 | 0.10ms | ~0.01ms | **~10x** |
| M4 六边形注意力 | 0.21ms | ~0.02ms | **~10x** |
| M5 校正 | 0.08ms | ~0.01ms | **~8x** |
| M6 全流水线 | 0.79ms | ~0.08ms | **~10x** |

> 注：上述为预期值，实际性能以昇腾环境测试为准。

## 四、API 服务部署

提供 FastAPI 推理服务，支持 HTTP 请求调用太极矩阵流水线：

```bash
# 启动 API 服务
python -m taichi_matrix.api_server --host 0.0.0.0 --port 8000

# 请求示例
curl -X POST http://localhost:8000/pipeline \
  -H "Content-Type: application/json" \
  -d '{"input": [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]]}'
```

## 五、集成到昇腾生态

### 5.1 昇腾社区发布
- 在 [昇腾社区](https://www.hiascend.com/developer/ascendhub) 发布镜像
- 参与昇腾开发者认证

### 5.2 华为云 ModelArts
- 发布为 ModelArts 自定义算法
- 支持一键训练/推理部署

### 5.3 与 MindSpore 深度适配（进阶）
- 将核心 C6 算子用 MindSpore 重写
- 深度利用昇腾硬件特性（AIGC 场景优化）

## 六、联系人

- 项目作者：孙永吉
- 太极矩阵社区：https://gitee.com/sun-yongji-yuyubenyuan_admin/taichi-matrix
