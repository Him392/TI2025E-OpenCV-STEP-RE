# TI2025E-OpenCV-STEP-RE

基于 OpenCV + YOLO 的步进电机目标追踪平台，通过串口控制步进电机实现实时目标跟踪。

## 主要功能

- **双模式检测**：传统 CV（空白检测）/ YOLO 深度学习检测，可切换
- **实时目标追踪**：从摄像头画面中检测目标，计算偏移向量
- **步进电机控制**：通过串口（RS485）发送指令控制 pan/tilt 两轴电机
- **OpenCV 可视化调参面板**：实时调节检测参数（阈值、置信度等）
- **激光偏移校准**：支持激光瞄准点微调

## 目录结构

```
├── main_new.py              # ⭐ 主程序入口（检测 + 追踪 + 电机控制）
├── blank_dect.py            # 空白检测模块（Canny + 轮廓逼近）
├── yolo_dect.py             # YOLO 检测模块（ultralytics）
├── Serial_STEP.py           # 串口步进电机通信协议
├── blank_dect_newdemo.py    # 胶带边界检测 GUI 调试工具
├── yolo_demo.py             # YOLO 检测演示脚本
├── train_yolo.py            # YOLO 模型训练脚本
├── take_pic.py              # 摄像头自动抓拍工具
├── best-bad.pt              # 当前使用的 YOLO 模型权重
├── best.pt                  # 备用模型权重
└── .gitignore
```

## 依赖环境

- Python 3.10+
- Windows（串口通信）

### Python 包

```bash
pip install opencv-python numpy pyserial ultralytics pillow
```

## 运行方式

### 主程序（检测 + 追踪 + 电机控制）

```bash
python main_new.py
```

在 `main_new.py` 顶部可修改关键参数：

```python
detection_method = 'yolo'     # 'yolo' 或 'blank'
camera_index = 1              # 摄像头编号
yolo_model_path = "best-bad.pt"  # 模型权重路径
laser_offset_x = -14          # 激光 X 偏移
laser_offset_y = -24          # 激光 Y 偏移
```

操作按键：
- `Space` — 启动/暂停追踪
- `Q` — 退出

### 其他独立工具

```bash
python blank_dect_newdemo.py  # 胶带边界检测 GUI 调试
python yolo_demo.py           # YOLO 检测演示
python train_yolo.py          # 模型训练（需修改内部路径）
python take_pic.py            # 摄像头自动抓拍
```

## 串口通信协议

步进电机通过 RS485 串口通信，协议格式（13 字节）：

| 字节 | 内容 | 说明 |
|------|------|------|
| 1 | 0x01 | 地址 |
| 2 | 0xF6 / 0xFD | 功能码（速度/位置模式） |
| 3 | 0x00 / 0x01 | 方向 |
| 4-5 | speed | 速度（最大 0x0BB8） |
| 6 | accel | 加速度（推荐 252） |
| 7-10 | pulses | 脉冲数（仅位置模式） |
| 11 | mode | 模式 |
| 12 | sync | 同步标志 |
| 13 | 0x6B | 校验位 |

默认串口：
- Pan 轴：`COM9`，115200 baud
- Tilt 轴：`COM10`，115200 baud

## 注意事项

- 串口号（`COM9`/`COM10`）请根据设备管理器实际端口修改（`Serial_STEP.py`）
- 电机方向（`0x00`/`0x01`）若与实际运动方向相反，互换即可
- YOLO 模型需使用 ultralytics 训练的 `.pt` 权重文件
- 摄像头索引从 0 开始，根据实际设备调整

---
如有问题或建议，欢迎 issue 反馈。
