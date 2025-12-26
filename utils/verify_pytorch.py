import torch
import torchvision
import torchaudio

print("=" * 50)
print("PyTorch 安装验证")
print("=" * 50)

print(f"\nPyTorch 版本: {torch.__version__}")
print(f"Torchvision 版本: {torchvision.__version__}")
print(f"Torchaudio 版本: {torchaudio.__version__}")

print("\n" + "=" * 50)
print("CUDA 支持检查")
print("=" * 50)

cuda_available = torch.cuda.is_available()
print(f"\nCUDA 可用: {cuda_available}")

if cuda_available:
    print(f"CUDA 版本: {torch.version.cuda}")
    print(f"cuDNN 版本: {torch.backends.cudnn.version()}")
    print(f"GPU 设备数量: {torch.cuda.device_count()}")
    print(f"当前 GPU 设备: {torch.cuda.current_device()}")
    print(f"GPU 设备名称: {torch.cuda.get_device_name(0)}")
    print(f"GPU 内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # 测试 GPU 计算
    print("\n" + "=" * 50)
    print("GPU 计算测试")
    print("=" * 50)
    try:
        x = torch.randn(3, 3).cuda()
        y = torch.randn(3, 3).cuda()
        z = torch.matmul(x, y)
        print("✓ GPU 计算测试成功！")
        print(f"  测试张量形状: {z.shape}")
    except Exception as e:
        print(f"✗ GPU 计算测试失败: {e}")
else:
    print("\n⚠ 警告: CUDA 不可用，PyTorch 将使用 CPU 模式")

print("\n" + "=" * 50)
print("验证完成")
print("=" * 50)

