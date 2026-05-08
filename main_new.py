from blank_dect import BlankDetector, DetectorControlPanel, open_configured_camera
import Serial_STEP
import cv2

# 摄像头编号：0 / 1 / 2 ... 按你的设备实际情况填写
camera_index = 1

# 摄像头分辨率与帧率（可修改）
camera_width = 252
camera_height = 288
camera_fps = 60

# 可视化参数面板初始值
blur_ksize = 3
canny_thresh_lower = 50
approx_ratio_x100 = 15
use_roi = 1

# 1. 实例化检测器
detector = BlankDetector(
    use_roi=bool(use_roi),
    blur_ksize=blur_ksize,
    canny_thresh_lower=canny_thresh_lower,
    approx_ratio=approx_ratio_x100 / 100.0,
)
# 2. 打开摄像头
cap = open_configured_camera(camera_index, camera_width, camera_height, camera_fps)
if cap is None:
    print(f"无法打开摄像头: {camera_index}")
    raise SystemExit(1)

control_panel = DetectorControlPanel(
    blur_ksize=blur_ksize,
    canny_thresh_lower=canny_thresh_lower,
    approx_ratio_x100=approx_ratio_x100,
    use_roi=use_roi,
)
control_panel.create()

cv2.namedWindow("Debug", cv2.WINDOW_NORMAL)

while True:
    control_panel.sync_to(detector)

    ret, frame = cap.read()
    if not ret: break
    
    # 3. 传入帧进行处理，获取偏置向量 (dx, dy) 和用于调试的图像
    vector, debug_img = detector.process_frame(frame, debug=True)
    
    if vector:
        print(f"追踪偏置: x={vector[0]}, y={vector[1]}")
    else:
        print("未检测到目标 / Target Lost")
        
    cv2.imshow("Debug", debug_img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
