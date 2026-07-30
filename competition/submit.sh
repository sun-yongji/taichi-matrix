#!/bin/bash
# 太极矩阵 · 昇腾算子大赛 · 提交包打包脚本
# 用法: bash submit.sh

set -e

NAME="taichi-hex-ascend"
VERSION="1.0.0"
OUTPUT="${NAME}-v${VERSION}.zip"

echo "📦 打包参赛作品: ${OUTPUT}"
echo ""

# 创建临时打包目录
TMPDIR=$(mktemp -d)
SUBDIR="${TMPDIR}/${NAME}"
mkdir -p "${SUBDIR}"

# 复制文件
cp -r ../competition/operator   "${SUBDIR}/"
cp -r ../competition/test       "${SUBDIR}/"
cp -r ../competition/doc        "${SUBDIR}/"
cp ../competition/README.md     "${SUBDIR}/"
cp ../COMPETITION.md            "${SUBDIR}/"

# 移除 __pycache__
find "${SUBDIR}" -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
find "${SUBDIR}" -name "*.pyc" -delete

# 创建压缩包
cd "${TMPDIR}"
zip -r "${OUTPUT}" "${NAME}/" > /dev/null
cp "${OUTPUT}" "${TMP_SRC}/competition/"
cd - > /dev/null

# 清理
rm -rf "${TMPDIR}"

echo "✅ 打包完成!"
echo "   文件: competition/${OUTPUT}"
echo "   大小: $(du -h competition/${OUTPUT} | cut -f1)"
echo ""
echo "📋 提交说明:"
echo "   1. 登录 https://www.hiascend.com/developer/contests/"
echo "   2. 上传 ${OUTPUT}"
echo "   3. 填写作品名称: 太极矩阵 — C6六边形注意力算子"
echo ""
