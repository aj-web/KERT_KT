#!/bin/bash
# KRD-KT 依赖安装脚本 (Bash)
# 用于快速修复依赖问题

echo "================================================================================"
echo "KRD-KT 依赖安装脚本"
echo "================================================================================"
echo ""

# 检查 Python 版本
echo "检查 Python 版本..."
python --version
echo ""

# 检查 CUDA
echo "检查 CUDA..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=driver_version --format=csv,noheader
else
    echo "  未检测到 NVIDIA GPU"
fi
echo ""

# 询问用户选择
echo "请选择 PyTorch 安装版本:"
echo "  1. CUDA 11.8"
echo "  2. CUDA 12.1"
echo "  3. CPU only"
echo ""
read -p "请输入选择 (1/2/3): " choice

# 安装 PyTorch
echo ""
echo "安装 PyTorch..."
case $choice in
    1)
        echo "  安装 CUDA 11.8 版本..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
        ;;
    2)
        echo "  安装 CUDA 12.1 版本..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
        ;;
    3)
        echo "  安装 CPU 版本..."
        pip install torch torchvision torchaudio
        ;;
    *)
        echo "  无效选择，默认安装 CUDA 11.8 版本..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
        ;;
esac

# 安装其他依赖
echo ""
echo "安装其他依赖..."
pip install numpy pandas scikit-learn tqdm matplotlib seaborn networkx

# 验证安装
echo ""
echo "验证安装..."
python -c "
import sys
try:
    import numpy as np
    print('✓ numpy:', np.__version__)
except ImportError as e:
    print('✗ numpy:', e)
    sys.exit(1)

try:
    import pandas as pd
    print('✓ pandas:', pd.__version__)
except ImportError as e:
    print('✗ pandas:', e)
    sys.exit(1)

try:
    import sklearn
    print('✓ scikit-learn:', sklearn.__version__)
except ImportError as e:
    print('✗ scikit-learn:', e)
    sys.exit(1)

try:
    import torch
    print('✓ torch:', torch.__version__)
    print('  CUDA available:', torch.cuda.is_available())
    if torch.cuda.is_available():
        print('  CUDA version:', torch.version.cuda)
except ImportError as e:
    print('✗ torch:', e)
    sys.exit(1)

try:
    import matplotlib
    print('✓ matplotlib:', matplotlib.__version__)
except ImportError as e:
    print('✗ matplotlib:', e)
    sys.exit(1)

try:
    import seaborn as sns
    print('✓ seaborn:', sns.__version__)
except ImportError as e:
    print('✗ seaborn:', e)
    sys.exit(1)

try:
    import networkx as nx
    print('✓ networkx:', nx.__version__)
except ImportError as e:
    print('✗ networkx:', e)
    sys.exit(1)

print('\n所有依赖已正确安装！')
"

if [ $? -eq 0 ]; then
    echo ""
    echo "================================================================================"
    echo "安装完成！"
    echo "================================================================================"
    echo ""
    echo "现在可以运行实验了："
    echo "  python experiments/core/run_experiment.py --dataset assist09 --mode sl --n_runs 1"
else
    echo ""
    echo "================================================================================"
    echo "安装失败！请检查错误信息"
    echo "================================================================================"
    echo ""
    echo "请参考 INSTALL.md 获取详细安装指南"
fi

