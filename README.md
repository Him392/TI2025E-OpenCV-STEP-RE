# TI2025E-OpenCV-STEP-RE

基于 **OpenCV、YOLO 和步进电机** 的视觉目标检测与双轴追踪项目。项目在原有视觉检测程序的基础上进行小范围修改，将执行机构改为通过串口控制的步进电机，实现目标检测、目标追踪、偏差计算以及 Pan/Tilt 双轴跟随。

> 当前仓库以 Windows + USB 摄像头 + 两路串口步进电机控制器为主要运行环境。

## 项目功能

- **YOLO 目标检测与追踪**
  - 使用 Ultralytics YOLO 加载本地 `.pt` 模型。
  - 使用 ByteTrack 持久化追踪目标 ID。
  - 支持置信度阈值实时调节。
- **传统 OpenCV 四边形检测（2025电赛E）**
  - 灰度化、中值滤波、Canny 边缘检测和多边形逼近。
  - 支持 ROI 区域跟踪、面积变化过滤和短时目标位置预测。
  - 适合检测具有明显矩形/四边形边界的目标，例如胶带或纸张边界。
- **双轴步进电机控制**
  - 根据目标中心与画面中心的像素偏差计算 Pan/Tilt 方向和速度。
  - 通过串口发送速度模式指令控制两路步进电机。
  - 目标丢失或程序退出时自动发送停止指令。
- **实时调试与参数调整**
  - OpenCV 控制面板可调节检测参数、YOLO 置信度和激光偏移量。
  - 实时显示检测框、目标中心、追踪向量和 ROI 区域。
- **辅助工具**
  - Tkinter 图形化的胶带边界检测调试工具。
  - YOLO 摄像头检测演示。
  - 摄像头定时抓拍工具。
  - YOLO 训练脚本示例。

## 文件说明

| 文件 | 作用 |
| --- | --- |
| `main_new.py` | 主程序：摄像头采集、检测/追踪、偏差计算和步进电机控制 |
| `blank_dect.py` | 基于 OpenCV 的四边形/空白区域检测模块，支持 ROI 和位置预测 |
| `yolo_dect.py` | 基于 Ultralytics YOLO + ByteTrack 的目标检测与追踪模块 |
| `Serial_STEP.py` | 两路步进电机串口连接和控制指令封装 |
| `blank_dect_newdemo.py` | Tkinter 胶带边界检测调试工具，可加载图片或打开摄像头 |
| `yolo_demo.py` | YOLO 检测/追踪演示程序，当前代码默认读取摄像头 1 |
| `take_pic.py` | 摄像头定时抓拍工具，默认每 1.5 秒保存一张图片 |
| `train_yolo.py` | YOLO 训练示例，需要根据本机环境修改模型和数据集路径 |
| `best-bad.pt` | 仓库内置 YOLO 模型权重，主程序当前默认使用 |
| `best.pt` | 备用 YOLO 模型权重 |
| `LICENSE` | GNU GPL v3 开源许可证 |

## 环境要求

- Python 3.10 或更高版本
- Windows（当前串口配置使用 `COM9` 和 `COM10`）
- USB 摄像头
- 两路可通过串口控制的步进电机驱动器
- 如运行 YOLO 推理，建议准备可用的 NVIDIA GPU；CPU 也可运行，但速度可能较慢

安装主要依赖：

```bash
pip install opencv-python numpy pyserial ultralytics pillow
```

其中：

- `opencv-python`：摄像头采集、图像处理和窗口显示
- `numpy`：数值计算
- `pyserial`：串口通信
- `ultralytics`：YOLO 检测与 ByteTrack 追踪
- `pillow`：Tkinter 调试工具中的图像显示

## 快速开始

### 1. 配置串口

编辑 `Serial_STEP.py`，将串口号修改为实际设备对应的端口：

```python
serial_pan_port = 'COM9'
serial_tilt_port = 'COM10'
```

当前文件在导入时会直接打开两路串口，因此运行主程序前必须确保端口存在且未被其他程序占用。

### 2. 配置主程序

编辑 `main_new.py` 顶部的参数：

```python
detection_method = 'yolo'    # 'yolo' 或 'blank'
camera_index = 1              # 摄像头编号
camera_width = 252
camera_height = 288
camera_fps = 60

yolo_model_path = 'best-bad.pt'
yolo_conf_thresh_x100 = 50

laser_offset_x = -14
laser_offset_y = -24
```

说明：

- `detection_method='yolo'`：使用 YOLO 模型检测和追踪。
- `detection_method='blank'`：使用传统 OpenCV 四边形检测。
- `camera_index`：摄像头索引从 `0` 开始，需要根据实际设备调整。
- `laser_offset_x`、`laser_offset_y`：用于校准摄像头中心与激光落点之间的偏移。
- `best-bad.pt` 和 `best.pt` 必须位于项目根目录，或填写正确的模型路径。

### 3. 运行主程序

```bash
python main_new.py
```

主程序启动后会打开摄像头和参数控制窗口：

- 按 `Space`：开始/暂停检测与追踪。
- 按 `Q`：退出程序。
- 检测窗口中的控制面板可实时调整 YOLO 置信度、激光偏移，或传统检测的滤波、Canny、逼近和 ROI 参数。

> 建议在未连接电机或首次调试时，先单独运行检测模块，确认摄像头、模型和偏差方向正确后再连接电机。

## 单独运行检测模块

### OpenCV 四边形检测

```bash
python blank_dect.py
```

也可以在其他 Python 程序中调用：

```python
from blank_dect import BlankDetector, open_configured_camera

cap = open_configured_camera(0, 640, 480, 30)
detector = BlankDetector(use_roi=True)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    vector, debug_img = detector.process_frame(frame, debug=True)
    print(vector)
```

### YOLO 检测与追踪

```bash
python yolo_dect.py
```

该模块默认加载当前目录下的 `best.pt`，如需使用其他权重，可以修改 `YoloDetector(model_path=...)` 的参数。

### 图形化传统检测调试工具

```bash
python blank_dect_newdemo.py
```

该工具支持：

- 选择图片进行检测；
- 打开摄像头实时检测；
- 调整中值滤波、Canny 阈值和多边形逼近参数；
- 查看原图、灰度/模糊图、边缘图和最终检测结果。

## 辅助工具

```bash
python yolo_demo.py    # 摄像头 YOLO 检测与追踪演示
python take_pic.py     # 摄像头定时抓拍
python train_yolo.py   # YOLO 训练示例
```

### `take_pic.py` 注意事项

当前脚本默认：

- 使用摄像头 `1`；
- 分辨率为 `252x288`；
- 图片保存到 `D:/yolo/takepic`；
- 每隔约 1.5 秒保存一张图片；
- 按空格暂停/继续，按 `Q` 或 `ESC` 退出。

如需在其他电脑运行，请先修改摄像头编号和保存目录。

### `train_yolo.py` 注意事项

当前训练脚本中的模型路径、数据集配置路径和训练输出目录是示例路径，包含 Windows 本地路径，例如：

```python
model = YOLO('D:/yolo/yolo11s.pt')
data='D:/yolo/dataset.yaml'
```

运行训练前需要根据本机目录、显卡和数据集配置进行修改。脚本默认使用较保守的 `batch=8`、`imgsz=320` 和 `workers=0` 参数。

## 步进电机控制说明

`main_new.py` 根据目标中心相对画面中心的偏差 `(dx, dy)` 计算两轴速度：

1. `dx` 控制 Pan 水平轴。
2. `dy` 控制 Tilt 俯仰轴。
3. 采用简单比例控制，并对速度进行限幅。
4. 偏差小于 5 像素时进入死区，减少电机抖动。
5. 没有检测到目标时，两路电机发送速度为 0 的停止指令。

`Serial_STEP.py` 使用的默认配置为：

- 地址：`0x01`
- 波特率：`115200`
- 数据格式：步进电机驱动器默认的 8N1 配置
- 校验/结束字节：`0x6B`
- 速度模式功能码：`0xF6`
- 位置模式功能码：`0xFD`

实际电机方向可能因安装方式和接线不同而相反。如果运动方向不正确，可在 `main_new.py` 中调整 `pan_dir`、`tilt_dir` 的方向映射，或交换 `0x00` 与 `0x01`。

## 常见问题

### 摄像头无法打开

- 检查 `camera_index` 是否正确；
- 确认摄像头未被其他程序占用；
- 检查 Windows 摄像头权限；
- 尝试使用 `0`、`1`、`2` 等索引逐一测试。

### YOLO 模型加载失败

- 确认 `.pt` 文件位于当前工作目录，或填写绝对路径；
- 确认已安装 `ultralytics`；
- 检查模型文件是否完整；
- 如果显存不足，可以降低输入尺寸、置信度或使用更小的模型。

### 串口打开失败

- 确认驱动器已连接并安装驱动；
- 在设备管理器中查看实际 COM 端口；
- 修改 `Serial_STEP.py` 中的端口号；
- 确认 COM 端口没有被串口调试工具占用。

### 电机运动方向错误或抖动

- 调整方向映射；
- 增大死区阈值；
- 降低比例控制系数或速度上限；
- 检查激光偏移量和摄像头安装位置；
- 首次调试时建议降低电机速度并断开机械负载。

## 项目状态

本项目是基于原项目进行的实验性修改版本，当前代码重点验证以下流程：

```text
摄像头采集
    ↓
YOLO / OpenCV 目标检测
    ↓
计算目标中心偏差
    ↓
串口发送 Pan/Tilt 步进电机速度指令
    ↓
实时调试显示与目标跟踪
```

代码中的摄像头编号、串口号、模型路径和训练路径均与开发环境相关，移植到其他设备时需要根据硬件和目录结构调整。

## 开源许可

本项目使用 [GNU General Public License v3.0](LICENSE) 发布。
