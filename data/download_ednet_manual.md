# EdNet数据集手动下载指南

## 方法1：百度网盘（推荐-国内最快）

EdNet数据集由于体积较大（>1GB），推荐使用以下方式：

1. **访问EdNet官方页面**
   - GitHub: https://github.com/riiid/ednet
   - 论文: https://arxiv.org/abs/1912.03072

2. **下载EdNet-KT1数据集**
   
   **选项A：直接下载（如果可用）**
   ```
   https://github.com/riiid/ednet/releases/download/v1.0/KT1.zip
   ```
   
   **选项B：使用Kaggle**
   - 访问: https://www.kaggle.com/c/riiid-test-answer-prediction
   - 注册Kaggle账号
   - 下载数据集
   
   **选项C：百度网盘/其他云盘**
   - 搜索"EdNet KT1 dataset"
   - 找到可靠的分享链接

3. **解压到指定目录**
   ```
   E:\IDAEWOREKSPACE\demo\data\raw\ednet\KT1\
   ```
   
   解压后目录结构应该是：
   ```
   data/raw/ednet/KT1/
   ├── questions.csv
   ├── lectures.csv
   └── u[学生ID].csv (多个学生交互文件)
   ```

## 方法2：使用EduData库（如果支持）

```bash
cd E:\IDAEWOREKSPACE\demo
python -c "from EduData import get_data; get_data('ednet', data_dir='data/raw/')"
```

## 方法3：Kaggle API（需要认证）

1. **安装Kaggle**
   ```bash
   pip install kaggle
   ```

2. **配置API密钥**
   - 访问: https://www.kaggle.com/settings
   - 创建API Token
   - 下载kaggle.json到 `~/.kaggle/` 目录

3. **下载数据**
   ```bash
   kaggle competitions download -c riiid-test-answer-prediction
   ```

## 验证下载

下载完成后，运行以下命令验证：

```bash
python data/process_ednet.py
```

如果看到数据统计信息（而不是错误），说明下载成功！

## 数据集大小参考

- **KT1.zip**: ~1.2 GB（压缩后）
- **解压后**: ~3.5 GB
- **处理后pkl**: ~500 MB

## 问题排查

### 问题1：下载速度慢
- 使用国内镜像或云盘分享
- 使用下载工具（IDM、迅雷等）

### 问题2：解压失败
- 检查磁盘空间（需要至少5GB）
- 使用7-Zip或WinRAR解压

### 问题3：文件损坏
- 重新下载
- 验证MD5/SHA1校验和

---

**当前状态**：等待EdNet数据下载完成

**替代方案**：可以先使用ASSIST09和Junyi进行模型开发和实验，EdNet留待后续大规模实验使用。

