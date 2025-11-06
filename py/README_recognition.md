# 车牌检测与识别系统使用指南

## 系统架构

```
原始图像 → YOLOv8检测 → 车牌裁剪 → 倾斜校正 → PaddleOCR识别 → 车牌号码
```

## 安装步骤

### 1. 激活你的conda环境

```bash
conda activate bighomework
```

### 2. 安装PaddleOCR（CPU版本）

```bash
# 安装PaddlePaddle（CPU版本）
pip install paddlepaddle

# 安装PaddleOCR
pip install paddleocr

# 安装其他依赖（如果需要）
pip install opencv-python shapely
```

**注意事项：**
- 首次运行会自动下载OCR模型（约50-100MB）
- 确保有网络连接
- 下载后的模型会缓存到 `~/.paddleocr/` 目录

### 3. 验证安装

```bash
python -c "from paddleocr import PaddleOCR; print('安装成功')"
```

## 文件说明

### 核心模块

1. **detect_specialist.py** - 车牌检测器
   - 使用YOLOv8检测车牌位置
   - 裁剪和倾斜校正

2. **plate_recognizer.py** - 字符识别器（新增）
   - 使用PaddleOCR识别车牌字符
   - 字符清洗和验证

3. **detect_and_recognize.py** - 端到端系统（新增）
   - 整合检测和识别
   - 批量处理功能

4. **ccpd_decoder.py** - CCPD数据集解码工具
   - 从文件名解析真实车牌号
   - 用于验证识别准确率

## 使用方法

### 方式1: 测试单张图像（快速验证）

```python
from detect_and_recognize import LicensePlateSystem

# 初始化系统
system = LicensePlateSystem(
    detector_model_path='runs/detect/yolov8n_finetune_on_ccpd/weights/best.pt',
    use_gpu_for_ocr=False  # CPU模式
)

# 处理单张图像
results, viz_image = system.process_single_image(
    'test_image.jpg',
    visualize=True
)

# 打印结果
for i, result in enumerate(results):
    print(f"车牌 {i+1}: {result['plate_number']}")
    print(f"  检测置信度: {result['detection_confidence']:.2%}")
    print(f"  识别置信度: {result['recognition_confidence']:.2%}")
```

### 方式2: 批量处理（推荐）

直接运行端到端脚本：

```bash
cd /mnt/nas-new/home/yangjiayuan/mypart/py
python detect_and_recognize.py
```

**修改配置（在 detect_and_recognize.py 中）：**

```python
# 模型路径
MODEL_PATH = 'runs/detect/yolov8n_finetune_on_ccpd/weights/best.pt'

# 输入输出路径
INPUT_FOLDER = 'CCPD2020/ccpd_green/images/test'  # 修改为你的测试集路径
OUTPUT_FOLDER = 'plate_recognition_results'        # 输出目录

# 是否使用GPU进行OCR
USE_GPU_FOR_OCR = False  # 如果有GPU可设为True

# 检测置信度阈值
CONFIDENCE_THRESHOLD = 0.5
```

### 方式3: 仅测试识别模块

测试PaddleOCR是否正常工作：

```bash
python plate_recognizer.py
```

## 输出结果

运行后会在输出文件夹生成：

```
plate_recognition_results/
├── visualizations/          # 可视化结果（带车牌号标注）
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── cropped_plates/          # 裁剪的车牌图像
│   ├── image1_plate_1.jpg
│   ├── image1_plate_2.jpg
│   └── ...
└── recognition_results.txt  # 详细识别结果
```

**recognition_results.txt 格式：**

```
车牌识别详细结果
================================================================================

文件: test_image_001.jpg
  车牌 1: 皖A12345
  检测置信度: 0.956
  识别置信度: 0.923
  位置: (308, 507, 524, 618)
--------------------------------------------------------------------------------
...
```

## 常见问题

### Q1: 安装PaddleOCR时报错

**问题**: `ERROR: Could not find a version that satisfies the requirement paddlepaddle`

**解决**:
```bash
# 尝试指定版本
pip install paddlepaddle==2.5.1

# 或使用清华镜像
pip install paddlepaddle -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### Q2: 首次运行很慢

**原因**: 正在下载OCR模型

**解决**: 等待模型下载完成（仅首次运行需要），后续会直接使用缓存

### Q3: 识别准确率不高

**可能原因**:
1. 车牌图像质量差（模糊、角度倾斜）
2. 光照条件不好
3. 车牌尺寸太小

**优化建议**:
- 调整 `_preprocess_plate()` 中的预处理参数
- 在 `detect_and_recognize.py` 中降低 `CONFIDENCE_THRESHOLD`
- 使用更高分辨率的输入图像

### Q4: 内存不足

**解决**:
```python
# 在 detect_and_recognize.py 中分批处理
# 修改 process_batch() 增加批次限制
```

## 性能参考

在你的Ubuntu 20.04系统上（CPU模式）：

- **检测速度**: ~50ms/图像（YOLOv8）
- **识别速度**: ~500ms/车牌（PaddleOCR）
- **总处理时间**: ~600ms/图像（包含1个车牌）

如果启用GPU，OCR速度可提升至 ~100ms/车牌。

## 下一步优化

1. **提高识别准确率**:
   - 在CCPD数据集上微调OCR模型
   - 训练专用的字符识别CNN

2. **提高处理速度**:
   - 使用GPU加速
   - 批量处理优化

3. **增强功能**:
   - 添加车牌追踪（视频）
   - 支持多类型车牌（双层、警用等）

## 技术支持

如有问题，请检查：
1. Python版本 ≥ 3.8
2. 所有依赖已正确安装
3. 模型文件路径正确
4. 输入图像格式支持（jpg/png/bmp）
