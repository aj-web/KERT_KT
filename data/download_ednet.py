"""
EdNet数据集下载脚本

EdNet是一个大规模层级化教育数据集，包含数十万学生的作答记录。
官方仓库：https://github.com/riiid/ednet

使用方法：
    python data/download_ednet.py --output_dir data/raw/ednet
"""

import os
import sys
import argparse
import requests
from tqdm import tqdm
import zipfile


def download_file(url, output_path):
    """
    下载文件并显示进度条
    
    Args:
        url: 下载链接
        output_path: 输出文件路径
    """
    print(f"正在下载: {url}")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(output_path, 'wb') as f, tqdm(
        desc=output_path,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for data in response.iter_content(chunk_size=1024):
            size = f.write(data)
            pbar.update(size)


def extract_zip(zip_path, extract_to):
    """
    解压ZIP文件
    
    Args:
        zip_path: ZIP文件路径
        extract_to: 解压目标目录
    """
    print(f"正在解压: {zip_path}")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"解压完成: {extract_to}")


def download_ednet(output_dir):
    """
    下载EdNet数据集
    
    Args:
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # EdNet-KT1 数据集链接
    # 注意：这些链接可能需要根据实际情况更新
    urls = {
        'kt1': 'https://github.com/riiid/ednet/releases/download/v1.0/KT1.zip',
        'questions': 'https://github.com/riiid/ednet/raw/master/data/KT1/questions.csv',
        'lectures': 'https://github.com/riiid/ednet/raw/master/data/KT1/lectures.csv',
    }
    
    print("="*50)
    print("EdNet数据集下载")
    print("="*50)
    print(f"输出目录: {output_dir}")
    print()
    
    # 下载主数据文件
    kt1_zip = os.path.join(output_dir, 'KT1.zip')
    if not os.path.exists(kt1_zip):
        try:
            download_file(urls['kt1'], kt1_zip)
        except Exception as e:
            print(f"下载失败: {e}")
            print("\n请手动下载EdNet数据集：")
            print("1. 访问: https://github.com/riiid/ednet")
            print("2. 下载 KT1.zip 文件")
            print(f"3. 将文件放置在: {output_dir}")
            return False
    else:
        print(f"文件已存在: {kt1_zip}")
    
    # 解压数据
    kt1_dir = os.path.join(output_dir, 'KT1')
    if not os.path.exists(kt1_dir):
        extract_zip(kt1_zip, output_dir)
    else:
        print(f"目录已存在: {kt1_dir}")
    
    print("\n" + "="*50)
    print("下载完成！")
    print("="*50)
    print(f"数据位置: {kt1_dir}")
    print("\n下一步:")
    print("运行预处理脚本: python data/process_ednet.py")
    
    return True


def main():
    parser = argparse.ArgumentParser(description='下载EdNet数据集')
    parser.add_argument('--output_dir', type=str, default='data/raw/ednet',
                        help='输出目录 (默认: data/raw/ednet)')
    
    args = parser.parse_args()
    
    success = download_ednet(args.output_dir)
    
    if not success:
        sys.exit(1)


if __name__ == '__main__':
    main()

