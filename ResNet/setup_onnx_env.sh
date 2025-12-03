#!/bin/bash
set -e

# 配置参数
VENV_NAME="onnx_fix_env"
PYTHON_CMD="python3"
VENV_PATH="${PWD}/${VENV_NAME}"
DEPS=(
  "onnx==1.19.1"
  "numpy==1.26.4"
  "decorator==5.1.1"
  "sympy==1.12"
  "attrs==23.2.0"
  "psutil==5.9.8"
  "cloudpickle==2.2.1"
  "scipy==1.11.4"
  "tornado==6.3.3"
)
PIP_MIRROR="https://repo.huaweicloud.com/repository/pypi/simple/"

# 1. 检查Python
echo "=== 检查Python ==="
command -v $PYTHON_CMD >/dev/null 2>&1 || { echo "错误：未找到Python3"; exit 1; }
echo "Python版本：$($PYTHON_CMD --version | awk '{print $2}')"

# 2. 重建虚拟环境
echo -e "\n=== 配置虚拟环境 ==="
[ -d "$VENV_PATH" ] && rm -rf "$VENV_PATH"
$PYTHON_CMD -m venv "$VENV_PATH" && source "${VENV_PATH}/bin/activate"
echo "虚拟环境激活：($VENV_NAME)"

# 3. 安装依赖
echo -e "\n=== 安装依赖 ==="
pip install --upgrade pip -i $PIP_MIRROR
pip install "${DEPS[@]}" -i $PIP_MIRROR

# 4. 验证
echo -e "\n=== 验证环境 ==="
pip list | grep -E "onnx|numpy|decorator|sympy"
$PYTHON_CMD -c "import onnx; import numpy" >/dev/null 2>&1 && echo "✅ 环境可用" || { echo "❌ 环境异常"; exit 1; }

# 使用说明
echo -e "\n=== 使用说明 ==="
echo "激活环境：source ${VENV_PATH}/bin/activate"
echo "运行修复：python 修复脚本.py"
echo "退出环境：deactivate"
echo "删除环境：rm -rf $VENV_PATH"
