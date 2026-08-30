# 贡献指南 · TaiChi Matrix

感谢你对 TaiChi Matrix 的关注！本文档说明如何参与TaiChi Matrix 统一工具链的开发与改进。

## 快速开始

1. Fork 本仓库
2. 安装依赖：`pip install -e ".[all,dev]"`
3. 运行测试：`pytest tests/`
4. 开始开发

## 贡献类型

### 核心模块开发
- Router（M1）：路由算法优化
- MTP（M2）：多 token 预测改进
- Quant（M3）：量化策略优化
- HexAttn（M4）：注意力机制改进
- Correct（M5）：误差校正算法

### 集成与工具链
- Pipeline 集成优化
- 新框架适配（JAX、TensorFlow 等）
- 性能基准测试工具

### 测试
- 各模块边界条件测试
- 端到端集成测试
- 性能回归测试

### 文档
- 技术白皮书更新
- API 文档完善
- 使用教程与示例

## 开发流程

1. Fork → 创建分支：`git checkout -b feat/mX-your-feature`
2. 编写代码与测试
3. 确保全部测试通过：`pytest tests/ -v`
4. 提交：`git commit -m "feat(mX): add xxx"`
5. 推送 → 提交 Pull Request

## Commit 规范

```
<type>(scope): <description>
```

| type | 用途 | scope |
|------|------|-------|
| feat | 新功能 | m1~m6, pipeline, docs |
| fix | 修复 bug | m1~m6, pipeline |
| perf | 性能优化 | m1~m6, pipeline |
| test | 测试 | m1~m6, integration |
| docs | 文档 | readme, whitepaper, api |
| refactor | 重构 | m1~m6, pipeline |

## 代码规范

- 遵循 PEP 8
- 函数添加 docstring
- 公开 API 添加类型注解
- 新功能必须附带测试
- 保持 159/159 测试全通过

## 测试要求

- 单模块测试：`pytest tests/test_xxx.py -v`
- 全量测试：`pytest tests/ -v`
- 覆盖率不低于现有水平

## TaiChi Matrix 工具链体系

| 站 | 仓库 | 功能 |
|----|------|------|
| M1 | taichi-router | MoE 动态路由 |
| M2 | taichi-mtp | 多 token 预测 |
| M3 | taichi-quant | 熵量化 |
| M4 | taichi-hex | 六边形注意力 |
| M5 | taichi-correct | 共识校正 |
| **M6** | **taichi-matrix** | 统一入口 |

## 联系方式

- Issue：在本仓库提交 Issue
- 邮箱：okskill@foxmail.com
- 社区：[易宇社区](https://gitee.com/yi-yu-community)

## 许可证

Apache-2.0
