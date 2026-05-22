"""
YoloDetector 模块使用例 / Usage Example:

from yolo_dect import YoloDetector, YoloDetectorControlPanel, open_configured_camera
import cv2

# 摄像头与参数（可修改）
camera_index = 0
camera_width = 640
camera_height = 480
camera_fps = 30

# 1. 实例化检测器 (默认模型为当前目录下的 best.pt)
detector = YoloDetector(model_path="best.pt", conf_thresh=0.5)

# 2. 打开摄像头
cap = open_configured_camera(camera_index, camera_width, camera_height, camera_fps)
if cap is None:
    print(f"无法打开摄像头: {camera_index}")
    raise SystemExit(1)

# 可视化参数面板（可选）
panel = YoloDetectorControlPanel(conf_thresh_x100=50, laser_x=0, laser_y=0)
panel.create()

while True:
    # 从面板同步参数到检测器
    panel.sync_to(detector)

    ret, frame = cap.read()
    if not ret:
        break

    # 处理帧并获取偏置向量
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
"""

import sys
import cv2
import numpy as np
import os
from ultralytics import YOLO

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


def open_configured_camera(camera_index=0, width=640, height=480, fps=30, print_info=True):
    """打开摄像头并尝试设置分辨率与帧率。"""
    cap = cv2.VideoCapture(camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)

    if not cap.isOpened():
        return None

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    if print_info:
        print(f"Camera opened: index={camera_index}, resolution={actual_w}x{actual_h}, fps={actual_fps}")
    return cap


class YoloDetectorControlPanel:
    """YOLO OpenCV 可视化参数调节面板。"""

    def __init__(self, conf_thresh_x100=50, laser_x=0, laser_y=0, window_name="YOLO Controls"):
        self.window_name = window_name
        self.conf_thresh_x100 = conf_thresh_x100
        self.laser_x_offset = laser_x
        self.laser_y_offset = laser_y

    @staticmethod
    def _noop(_value):
        return

    def create(self):
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 420, 220)
        cv2.createTrackbar("Conf Thresh", self.window_name, self.conf_thresh_x100, 100, self._noop)
        cv2.createTrackbar("Laser X", self.window_name, self.laser_x_offset + 100, 200, self._noop)  # 100 is center (+/- 100)
        cv2.createTrackbar("Laser Y", self.window_name, self.laser_y_offset + 100, 200, self._noop)  # 100 is center (+/- 100)

    def sync_to(self, detector):
        current_conf = cv2.getTrackbarPos("Conf Thresh", self.window_name)
        laser_x = cv2.getTrackbarPos("Laser X", self.window_name) - 100
        laser_y = cv2.getTrackbarPos("Laser Y", self.window_name) - 100

        detector.conf_thresh = max(0.01, current_conf / 100.0)
        detector.laser_offset_x = laser_x
        detector.laser_offset_y = laser_y

class YoloDetector:
    def __init__(self, model_path="best.pt", conf_thresh=0.5):
        if not os.path.exists(model_path):
            print(f"Warning: Model file {model_path} not found. Ensure 'best.pt' is in the same directory.")
            
        self.model = YOLO(model_path)
        self.conf_thresh = conf_thresh
        self.laser_offset_x = 0
        self.laser_offset_y = 0
        
        # Tracking states
        self.active_track_id = None
        self.track_lost_frames = 0

    def process_frame(self, frame, debug=False):
        """
        处理单帧图像。
        返回: (dx, dy), display_img
        如果未检测到且无法预测，(dx, dy) 返回 None
        """
        if frame is None:
            return None, None
            
        display_img = frame.copy()
        
        img_h, img_w = display_img.shape[:2]
        # 加入人工激光偏移量后的新目标中心点
        img_center_x = (img_w // 2) + self.laser_offset_x
        img_center_y = (img_h // 2) + self.laser_offset_y
        
        if debug:
            # 绘制带有偏移的虚拟中心准星（代表激光落点）
            cv2.line(display_img, (img_center_x - 10, img_center_y), (img_center_x + 10, img_center_y), (255, 0, 0), 1)
            cv2.line(display_img, (img_center_x, img_center_y - 10), (img_center_x, img_center_y + 10), (255, 0, 0), 1)
            cv2.circle(display_img, (img_center_x, img_center_y), 4, (255, 0, 0), -1)

        # YOLO tracking inference
        results = self.model.track(frame, conf=self.conf_thresh, persist=True, tracker="bytetrack.yaml", verbose=False)
        boxes = results[0].boxes
        
        target_box = None
        target_conf = 0.0
        
        # Check if we have tracking IDs
        if boxes is not None and len(boxes) > 0 and boxes.id is not None:
            ids = boxes.id.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            xyxys = boxes.xyxy.cpu().numpy()
            
            # 如果尚未锁定 ID，选取置信度最高的目标作为追踪目标
            if self.active_track_id is None:
                best_idx = int(np.argmax(confs))
                self.active_track_id = int(ids[best_idx])
                
            # 寻找当前帧中与锁定 ID 匹配的框
            if self.active_track_id is not None:
                match_indices = np.where(ids == self.active_track_id)[0]
                if len(match_indices) > 0:
                    idx = match_indices[0]
                    target_box = xyxys[idx]
                    target_conf = confs[idx]
                    self.track_lost_frames = 0
                else:
                    self.track_lost_frames += 1
        else:
            # 没有检测到或者没有分配到 ID
            if self.active_track_id is not None:
                self.track_lost_frames += 1
                
        # 丢失超时处理
        if self.track_lost_frames >= 10:  # 容忍丢失10帧
            self.active_track_id = None
            
        delta_vector = None
        
        if target_box is not None:
            x1, y1, x2, y2 = map(int, target_box)
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            
            dx = cx - img_center_x
            dy = cy - img_center_y
            delta_vector = (dx, dy)
            
            if debug:
                # 绘制追踪框
                cv2.rectangle(display_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                for pt in [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]:
                    cv2.circle(display_img, pt, 4, (0, 0, 255), -1)
                
                # 绘制置信度和ID
                text = f"ID: {self.active_track_id} Conf: {target_conf:.2f}"
                cv2.putText(display_img, text, (x1, max(10, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # 绘制中心准星连线
                cv2.circle(display_img, (cx, cy), 4, (0, 255, 255), -1)
                cv2.arrowedLine(display_img, (img_center_x, img_center_y), (cx, cy), (255, 0, 255), 2, tipLength=0.08)
                cv2.putText(display_img, f"Track Vector ({dx} px, {dy} px)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        elif debug and self.active_track_id is not None:
            cv2.putText(display_img, f"Tracking ID {self.active_track_id} LOST! ({self.track_lost_frames}/10)", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

        return delta_vector, display_img


def run_camera(camera_index=0, width=640, height=480, fps=30):
    """
    独立运行模块进行测试的函数模块，支持传入多个摄像头索引进行多摄选择测试
    """
    try:
        detector = YoloDetector(model_path="best.pt", conf_thresh=0.5)
    except Exception as e:
        print(f"初始化 YOLO 失败: {e}")
        return

    cap = open_configured_camera(camera_index, width, height, fps, print_info=False)

    if cap is None:
        print(f"无法打开摄像头: {camera_index}")
        return

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"摄像头 {camera_index} 已开启，分辨率={actual_w}x{actual_h}，fps={actual_fps}；按 'q' 退出")
    
    panel = YoloDetectorControlPanel()
    panel.create()
    
    while True:
        panel.sync_to(detector)
        
        ret, frame = cap.read()
        if not ret:
            print("读取摄像头帧失败")
            break
            
        h, w = frame.shape[:2]
        if max(h, w) > 1000:
            ratio = 1000.0 / max(h, w)
            frame = cv2.resize(frame, (int(w * ratio), int(h * ratio)))
            
        vector, res_img = detector.process_frame(frame, debug=True)
        
        cv2.imshow("YOLO Detector module", res_img)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    import sys
    
    cam_id = 0
    if len(sys.argv) > 1:
        try:
            cam_id = int(sys.argv[1])
        except ValueError:
            pass
            
    run_camera(cam_id)
