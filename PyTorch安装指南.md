# PyTorch 版本安装指南

## 概述

本指南将帮助你根据系统环境选择合适的 PyTorch 版本，并解决版本兼容性问题。

## 一、检查系统环境

### 1. 检查 CUDA 驱动版本

在 Windows PowerShell 中运行：

```powershell
nvidia-smi
```

查看输出中的 `CUDA Version`，例如：
```
CUDA Version: 12.7
```

**注意**：这个版本表示驱动支持的最高 CUDA 版本，通常可以运行更低版本的 CUDA（向后兼容）。

### 2. 检查已安装的 PyTorch 相关包

```powershell
pip show torch torchvision torchaudio
```

查看当前安装的版本，特别关注：
- `torch` 版本
- `torchvision` 版本（需要与 torch 匹配）
- `torchaudio` 版本（需要与 torch 匹配）

## 二、选择合适的 PyTorch 版本

### 版本兼容性规则

1. **torch、torchvision、torchaudio 必须版本匹配**
   - 例如：`torch 2.7.0+cu128` 需要 `torchvision` 和 `torchaudio` 也支持 CUDA 12.8

2. **CUDA 版本选择**
   - 如果驱动支持 CUDA 12.7，可以安装 CUDA 12.8 的 PyTorch（向后兼容）
   - 如果驱动支持 CUDA 11.8，可以安装 CUDA 11.8 的 PyTorch

3. **CPU 版本 vs GPU 版本**
   - 有 NVIDIA GPU：选择带 CUDA 后缀的版本（如 `2.7.0+cu128`）
   - 无 GPU 或仅 CPU 训练：选择 CPU 版本（如 `2.7.0+cpu`）

## 三、安装步骤

### 方法 1：使用官方安装命令（推荐）

访问 [PyTorch 官网](https://pytorch.org/get-started/locally/)，根据你的配置生成安装命令。

例如，对于 CUDA 12.8：
```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

### 方法 2：在 requirements.txt 中指定版本

在 `requirements.txt` 文件中添加：

```txt
torch==2.7.0+cu128
torchvision==0.22.0+cu128
torchaudio==2.7.0+cu128
```

然后安装：
```powershell
pip install -r requirements.txt
```

### 方法 3：仅安装 torch（让 pip 自动处理依赖）

```powershell
pip install torch==2.7.0+cu128
```

pip 会自动安装兼容的 `torchvision` 和 `torchaudio`。

## 四、常见问题解决

### 问题 1：版本冲突

**错误信息**：
```
torchvision 0.22.0+cu128 requires torch==2.7.0+cu128, but you have torch 2.6.0 which is incompatible.
```

**解决方法**：
1. 卸载冲突的包：
   ```powershell
   pip uninstall torch torchvision torchaudio
   ```
2. 重新安装匹配的版本：
   ```powershell
   pip install torch==2.7.0+cu128 torchvision==0.22.0+cu128 torchaudio==2.7.0+cu128
   ```

### 问题 2：CUDA 版本不匹配

**检查方法**：
```python
import torch
print(torch.cuda.is_available())  # 应该返回 True
print(torch.version.cuda)  # 查看 PyTorch 使用的 CUDA 版本
```

**解决方法**：
- 如果 `torch.cuda.is_available()` 返回 `False`，可能需要：
  1. 安装匹配的 CUDA 版本
  2. 或者安装 CPU 版本（如果不需要 GPU）

### 问题 3：requirements.txt 中有多个 torch 版本

**错误示例**：
```txt
torch==2.6.0+cu126
torch==2.7.1
torch==2.5.1
```

**解决方法**：
- 只保留一个版本，删除其他重复项
- 选择与系统中 `torchvision` 和 `torchaudio` 兼容的版本

## 五、验证安装

安装完成后，运行以下 Python 代码验证：

```python
import torch
import torchvision
import torchaudio

print(f"PyTorch 版本: {torch.__version__}")
print(f"Torchvision 版本: {torchvision.__version__}")
print(f"Torchaudio 版本: {torchaudio.__version__}")
print(f"CUDA 可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA 版本: {torch.version.cuda}")
    print(f"GPU 设备: {torch.cuda.get_device_name(0)}")
```

## 六、版本对照表

### CUDA 12.8 系列
- `torch==2.7.0+cu128`
- `torchvision==0.22.0+cu128`
- `torchaudio==2.7.0+cu128`

### CUDA 12.1 系列
- `torch==2.1.0+cu121`
- `torchvision==0.16.0+cu121`
- `torchaudio==2.1.0+cu121`

### CUDA 11.8 系列
- `torch==2.0.0+cu118`
- `torchvision==0.15.0+cu118`
- `torchaudio==2.0.0+cu118`

### CPU 版本
- `torch==2.7.0+cpu`
- `torchvision==0.22.0+cpu`
- `torchaudio==2.7.0+cpu`

**注意**：版本对照表会随 PyTorch 更新而变化，建议访问 [PyTorch 官网](https://pytorch.org/get-started/previous-versions/) 查看最新版本信息。

## 七、最佳实践

1. **统一管理版本**：在 `requirements.txt` 中明确指定所有 PyTorch 相关包的版本
2. **定期更新**：定期检查并更新到稳定版本
3. **环境隔离**：使用虚拟环境（conda 或 venv）避免版本冲突
4. **测试验证**：安装后运行简单测试确保 GPU 可用（如果使用 GPU）

## 八、快速检查清单

在安装前，确认以下信息：
- [ ] CUDA 驱动版本（`nvidia-smi`）
- [ ] 已安装的 PyTorch 版本（`pip show torch`）
- [ ] 项目需要的 PyTorch 版本
- [ ] 是否需要 GPU 支持
- [ ] `requirements.txt` 中是否有版本冲突

---

**最后更新**：2025-12-25

