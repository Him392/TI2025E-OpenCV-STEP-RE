import cv2

def clahe_preprocessing(frame):
    # 1. 将 BGR 图像转换到 YUV 色彩空间（或者 Lab 空间）
    # 这样可以只增强亮度通道(Y)，而不破坏原来的颜色通道(UV)
    yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
    
    # 2. 创建 CLAHE 对象（clipLimit是对比度限制，tileGridSize是网格大小）
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    
    # 3. 仅对 Y 通道（亮度通道）进行限制对比度自适应直方图均衡化
    yuv[:, :, 0] = clahe.apply(yuv[:, :, 0])
    
    # 4. 将 YUV 图像转回 BGR 空间
    result = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
    return result

import sys
import os
import tkinter as tk
from tkinter import filedialog

def main():
    if len(sys.argv) >= 2:
        image_path = sys.argv[1]
    else:
        # 如果没有通过命令行参数传入图片，则弹出文件选择对话框
        print("未检测到拖拽或命令行参数，准备弹出文件选择对话框...")
        root = tk.Tk()
        root.withdraw()  # 隐藏 tk 主窗口
        image_path = filedialog.askopenfilename(
            title="请选择要测试的图片",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.tiff"), ("All Files", "*.*")]
        )
        if not image_path:
            print("未选择任何图片，程序退出。")
            return

    if not os.path.exists(image_path):
        print(f"找不到图片: {image_path}")
        return

    # 读取图片
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"无法读取图片: {image_path}")
        return

    # 为了方便显示，可以等比例缩放图片，控制最大宽高
    max_dim = 800
    h, w = frame.shape[:2]
    if h > max_dim or w > max_dim:
        scale = max_dim / max(h, w)
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

    # 调用已有的 CLAHE 处理函数
    processed_frame = clahe_preprocessing(frame)

    # 拼接原图和处理后的图像进行直观对比
    combined = cv2.hconcat([frame, processed_frame])

    # 显示对比图像
    cv2.imshow("Original (Left) vs CLAHE (Right)", combined)
    print("按任意键退出...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

    
