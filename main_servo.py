import sys
import threading
import time
from collections import deque
import os

import cv2
if os.name == 'nt':
    import msvcrt
else:
    import select

""" 以下是自建包，存放于include目录下 """
from include.dect import RectangleDetector
from include.PWM import ServoController
from include.camera_reader import CameraReader
from include.pid import PID, PIDParams,PIDController
from include.display import DebugDisplay

tilt_value = 600  # 垂直舵机初始值

# 可调检测参数
class DetectionParams:
    def __init__(self):
        self.min_area = 5000
        self.min_rectangularity = 0.8
        self.max_aspect_ratio = 2.0
        self.distance_weight = 0.3
        self.adaptive_block_size = 11
        self.adaptive_c = 2
        self.morph_kernel_size = 3
        self.gaussian_blur_size = 5
        self.show_params = False

detection_params = DetectionParams()
pid_params = PIDParams()

# 全局变量
stop_sending = False
# kalman_filter = KalmanFilter(process_noise=1e-4, measurement_noise=1e-2)
prev_center = None
frame_queue = deque(maxlen=3)
trail_image = None
control_enabled = True  # 控制状态标志
center_stay_timer = 0    # 中心区域停留计时器
in_center_zone = False   # 是否在中心区域
fanzhuan = False  # 是否反转舵机方向

# 初始化舵机控制器
controller = ServoController()
# 设置舵机机初始位置
controller.servoset(servonum=3, angle=480)  # 水平舵机
controller.servoset(servonum=4, angle=700)  # 垂直舵机

# 初始化PID控制器
pan_pid = PID(
    p=pid_params.pan_kp, 
    i=pid_params.pan_ki, 
    d=pid_params.pan_kd, 
    imax=pid_params.pan_imax
)

tilt_pid = PID(
    p=pid_params.tilt_kp, 
    i=pid_params.tilt_ki, 
    d=pid_params.tilt_kd, 
    imax=pid_params.tilt_imax
)

def input_listener():
    """监听键盘输入"""
    global stop_sending
    while not stop_sending:
        try:
            if os.name == 'nt':
                if msvcrt.kbhit():
                    user_input = msvcrt.getwch()
                    if user_input.lower() == 'q':
                        stop_sending = True
                        print("\n停止程序...")
                time.sleep(0.05)
            else:
                if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
                    user_input = sys.stdin.read(1)
                    if user_input.lower() == 'q':
                        stop_sending = True
                        print("\n停止程序...")
        except Exception:
            break

def control_servos(pan_output, tilt_output, detected):
    """控制舵机运动，优化反转逻辑和边界处理"""
    try:
        global tilt_value
        global fanzhuan
        global controller
        # 舵机边界
        PAN_MIN, PAN_MAX = 128, 892
        TILT_MIN, TILT_MAX = 256, 700
        PAN_CENTER = 480
        PAN_STEP = 30
        # 未检测到目标时，自动扫描，方向可反转
        if not detected and control_enabled:
            pan_value = PAN_CENTER + (PAN_STEP if not fanzhuan else -PAN_STEP)
            # 垂直舵机保持当前值
        else:
            # 反转逻辑：pan_output正负方向取反
            pan_value = int(PAN_CENTER - pan_output * 0.5)
            pan_value = max(PAN_MIN, min(PAN_MAX, pan_value))
            tilt_value = int(tilt_value - tilt_output * 0.2)
            tilt_value = max(TILT_MIN, min(TILT_MAX, tilt_value))
        controller.servoset(servonum=3, angle=pan_value)
        controller.servoset(servonum=4, angle=tilt_value)
        print(f"舵机控制: 水平: {pan_value}, 垂直: {tilt_value}, 检测: {detected}, 反转: {fanzhuan}")
    except Exception as e:
        print(f"舵机控制错误: {e}")

# 启动键盘监听线程
try:
    input_thread = threading.Thread(target=input_listener)
    input_thread.daemon = True
    input_thread.start()
    print("按 'q' 停��程序")
except:
    print("无法启动输入监听线程")

# 使用CameraReader初始化摄像头
try:
    camera_reader = CameraReader(
        cam_id=0, 
        width=800, 
        height=600, 
        max_fps=60
    )
    print("摄像头初始化成功")
except Exception as e:
    print(f"摄像头初始化失败: {e}")
    exit()

# 性能监控
frame_count = 0
start_time = time.time()
fps = 0
processing_times = deque(maxlen=30)
send_counter = 0
send_interval = 2

# 参数调整步长
PARAM_STEP = {
    'min_area': 10,
    'min_rectangularity': 0.05,
    'max_aspect_ratio': 0.5,
    'distance_weight': 0.1,
    'adaptive_block_size': 2,
    'adaptive_c': 1,
    'morph_kernel_size': 1,
    'gaussian_blur_size': 2,
    'pan_kp': 0.002,
    'pan_ki': 0.001,
    'pan_kd': 0.005,
    'tilt_kp': 0.01,
    'tilt_ki': 0.001,
    'tilt_kd': 0.005,
    'output_scaler': 0.1
}

# 初始化多线程检测与PID控制类
rect_detector = RectangleDetector(detection_params)
rect_detector.start()
pid_controller = PIDController(pid_params)
pid_controller.start()
debug_display = DebugDisplay(detection_params, pid_params)

# 主循环
try:
    while not stop_sending:  # 循环直到收到停止信号
        # 从CameraReader获取帧
        retval, frame = camera_reader.read()
        if not retval:
            print("无法从摄像头读取帧，等待下一帧...")
            time.sleep(0.01)
            continue
        process_start = time.time()
        # 多线检测
        rect_detector.update_frame(frame)
        center_point, contour = rect_detector.get_result()
        filtered_point = center_point if center_point else None
        # 计算性能指标
        process_time = (time.time() - process_start) * 1000
        processing_times.append(process_time)
        avg_process_time = sum(processing_times) / len(processing_times) if processing_times else 0
        frame_count += 1
        elapsed_time = time.time() - start_time
        if elapsed_time > 1:
            fps = frame_count / elapsed_time
            frame_count = 0
            start_time = time.time()
        display_img = frame.copy()
        height, width = display_img.shape[:2]
        img_center = (width // 2, (height // 2) - 4)
        # 计算偏移量
        if filtered_point is not None:
            offset_x = filtered_point[0] - img_center[0]
            offset_y = filtered_point[1] - img_center[1]
        else:
            offset_x = 0
            offset_y = 0
        # 更新PID参数（热更新，确保PID对象参数同步）
        pid_controller.update_pid_params(pid_params)
        pid_controller.update_offset((offset_x, offset_y))
        pan_output, tilt_output = pid_controller.get_output()
        # 检查中心区域
        center_zone_size = 48
        in_center = filtered_point is not None and abs(offset_x) < center_zone_size and abs(offset_y) < center_zone_size
        if in_center:
            if not in_center_zone:
                in_center_zone = True
                center_stay_timer = time.time()
        else:
            in_center_zone = False
        debug_display.update(fps, avg_process_time, in_center_zone, center_stay_timer)
        display_img = debug_display.draw(display_img, img_center, filtered_point, contour, control_enabled)

        # 控制舵机运动
        if control_enabled:
            control_servos(pan_output, tilt_output, filtered_point is not None)
        else:
            control_servos(0, 0, False)

        # 打印调试信息
        print("\033c", end="")
        print(f"=== 实时检测结果 (FPS: {fps:.1f}) ===")
        if filtered_point is not None:
            print(f"矩形中心坐标: ({filtered_point[0]}, {filtered_point[1]})")
            print(f"中心偏移量: X: {offset_x}, Y: {offset_y}")
            if in_center_zone:
                stay_time = time.time() - center_stay_timer
                print(f"中心区域停留: {stay_time:.2f}s")
        print(f"控制状态: {'已启用' if control_enabled else '已禁用'}")
        print(f"处理延迟: {avg_process_time:.1f}ms")
        print("=" * 40)

        send_counter += 1
        if send_counter >= send_interval:
            send_counter = 0
except Exception as e:
    print(f"运行时错误: {e}")

finally:  # 收尾，确保资源释放
    rect_detector.stop()
    pid_controller.stop()
    # 停止摄像头
    if 'camera_reader' in locals():
        camera_reader.stop()
    
    cv2.destroyAllWindows()           
    # 释放舵机
    controller.servo_release(servonum=3)
    controller.servo_release(servonum=4)
    print("舵机已释放")

    print("程序已退出")
