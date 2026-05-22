from blank_dect import BlankDetector, DetectorControlPanel, open_configured_camera
from yolo_dect import YoloDetector, YoloDetectorControlPanel
import Serial_STEP
import cv2

# 检测方式选择: 'blank' 或 'yolo'
detection_method = 'yolo'

# 摄像头编号：0 / 1 / 2 ... 按你的设备实际情况填写
camera_index = 1

# 摄像头分辨率与帧率（可修改）
camera_width = 252
camera_height = 288
camera_fps = 60

# 可视化参数面板初始值 (针对 blank)
blur_ksize = 6
canny_thresh_lower = 170
approx_ratio_x100 = 5
use_roi = 1

# 可视化参数面板初始值 (针对 YOLO)
yolo_conf_thresh_x100 = 50
yolo_model_path = "best-bad.pt"

# 激光微调初始偏移量 (0 代表不偏移，负数靠左/上，正数靠右/下)
laser_offset_x = -14
laser_offset_y = -24

# 1. 实例化检测器
if detection_method == 'yolo':
    detector = YoloDetector(
        model_path=yolo_model_path,
        conf_thresh=yolo_conf_thresh_x100 / 100.0,
    )
    detector.laser_offset_x = laser_offset_x
    detector.laser_offset_y = laser_offset_y
else:
    detector = BlankDetector(
        use_roi=bool(use_roi),
        blur_ksize=blur_ksize,
        canny_thresh_lower=canny_thresh_lower,
        approx_ratio=approx_ratio_x100 / 100.0,
    )
    detector.laser_offset_x = laser_offset_x
    detector.laser_offset_y = laser_offset_y

# 2. 打开摄像头
cap = open_configured_camera(camera_index, camera_width, camera_height, camera_fps)
if cap is None:
    print(f"无法打开摄像头: {camera_index}")
    raise SystemExit(1)

if detection_method == 'yolo':
    control_panel = YoloDetectorControlPanel(
        conf_thresh_x100=yolo_conf_thresh_x100,
        laser_x=laser_offset_x,
        laser_y=laser_offset_y,
    )
else:
    control_panel = DetectorControlPanel(
        blur_ksize=blur_ksize,
        canny_thresh_lower=canny_thresh_lower,
        approx_ratio_x100=approx_ratio_x100,
        use_roi=use_roi,
        laser_x=laser_offset_x,
        laser_y=laser_offset_y,
    )
control_panel.create()

cv2.namedWindow("Debug", cv2.WINDOW_NORMAL)

is_detecting = False  # 空格键切换：是否开启识别/追踪

while True:
    control_panel.sync_to(detector)

    ret, frame = cap.read()
    if not ret: break
    
    # 3. 传入帧进行处理，获取偏置向量 (dx, dy) 和用于调试的图像
    if is_detecting:
        vector, debug_img = detector.process_frame(frame, debug=True)
    else:
        vector = None
        debug_img = frame.copy()
        cv2.putText(debug_img, "PAUSED - Press SPACE to start tracking", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
    
    if vector:
        dx, dy = vector[0], vector[1]
        print(f"追踪偏置: x={dx}, y={dy}")
        
        # --- 简单的比例(P)控制计算电机速度 ---
        # 屏幕宽度252的一半是126。让 dx=126 时达到最大理论平移速度 60rpm -> kp_pan ≈ 60/126 ≈ 0.48
        # 屏幕高度288的一半是144。让 dy=144 时达到最大理论俯仰速度 20rpm -> kp_tilt ≈ 20/144 ≈ 0.14
        
        pan_speed = int(abs(dx) * 0.48)
        tilt_speed = int(abs(dy) * 0.14)
        
        # 限幅
        pan_speed = min(40, max(0, pan_speed))
        tilt_speed = min(20, max(0, tilt_speed))
        
        # 确定方向（视实际电机接线，这里假设 dx>0->0x01, dx<0->0x00）
        # 画面向右偏(dx>0)，电机应该往右追；这里方向如果不对应，把 0x01 和 0x00 互换即可
        pan_dir = 0x01 if dx > 0 else 0x00
        tilt_dir = 0x01 if dy > 0 else 0x00
        
        # 若偏差极小则给死区，避免抖动
        if abs(dx) < 5: pan_speed = 0
        if abs(dy) < 5: tilt_speed = 0

        # 调用串口发送：采用 velocity (F6) 或连续相对位置(0x01)发大脉冲。
        Serial_STEP.send_speed(Serial_STEP.ser_pan, pan_dir, pan_speed, accel=252)
        Serial_STEP.send_speed(Serial_STEP.ser_tilt, tilt_dir, tilt_speed, accel=252)
        
    else:
        print("未检测到目标 / Target Lost")
        # 丢失目标，电机停止
        Serial_STEP.send_speed(Serial_STEP.ser_pan, 0x00, speed=0, accel=50)
        Serial_STEP.send_speed(Serial_STEP.ser_tilt, 0x00, speed=0, accel=50)
        
    cv2.imshow("Debug", debug_img)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord(' '):
        is_detecting = not is_detecting
        # 若是取消识别，清空可能锁定的 YOLO ID，确保下次开启时能重新索敌
        if not is_detecting and detection_method == 'yolo':
            detector.active_track_id = None

# 退出前停稳电机
Serial_STEP.send_speed(Serial_STEP.ser_pan, 0x00, speed=0, accel=50)
Serial_STEP.send_speed(Serial_STEP.ser_tilt, 0x00, speed=0, accel=50)

cap.release()
cv2.destroyAllWindows()
