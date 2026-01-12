#!/bin/bash
# Crazyflie悬停测试启动脚本

# 获取脚本所在目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"

echo "============================================================"
echo "Crazyflie 悬停测试 - 使用Crazyswarm"
echo "============================================================"
echo ""

# 检查是否安装了pycrazyswarm
if ! python3 -c "import pycrazyswarm" 2>/dev/null; then
    echo "❌ 错误: 未找到pycrazyswarm库"
    echo ""
    echo "请安装pycrazyswarm:"
    echo "  方式1: pip install pycrazyswarm"
    echo "  方式2: 从源码编译（推荐）"
    echo "         https://crazyswarm.readthedocs.io/"
    echo ""
    exit 1
fi

# 检查配置文件
CONFIG_DIR="$PROJECT_ROOT/aquila/config"
if [ ! -f "$CONFIG_DIR/crazyflies.yaml" ]; then
    echo "❌ 错误: 未找到配置文件 crazyflies.yaml"
    echo "   请在以下位置创建配置文件: $CONFIG_DIR/crazyflies.yaml"
    echo ""
    exit 1
fi

if [ ! -f "$CONFIG_DIR/launch.yaml" ]; then
    echo "❌ 错误: 未找到配置文件 launch.yaml"
    echo "   请在以下位置创建配置文件: $CONFIG_DIR/launch.yaml"
    echo ""
    exit 1
fi

# 设置配置文件路径（Crazyswarm会在当前目录或指定路径查找配置）
export CRAZYSWARM_CONFIG_PATH="$CONFIG_DIR"

echo "✅ 配置文件路径: $CONFIG_DIR"
echo "✅ Python环境检查通过"
echo ""

# 进入项目根目录
cd "$PROJECT_ROOT"

# 运行测试脚本
echo "启动悬停测试..."
echo ""
python3 aquila/scripts/test_hover_crazy.py

echo ""
echo "============================================================"
echo "测试结束"
echo "============================================================"
