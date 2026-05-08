"""
BlankDetector 模块使用例 / Usage Example (同步到当前 API):

from blank_dect import BlankDetector, DetectorControlPanel, open_configured_camera
import cv2

# 摄像头与参数（可修改）
camera_index = 0
camera_width = 640
camera_height = 480
camera_fps = 30

# 1. 实例化检测器
detector = BlankDetector(use_roi=True)

# 2. 打开摄像头（使用模块封装的 open_configured_camera）
cap = open_configured_camera(camera_index, camera_width, camera_height, camera_fps)
if cap is None:
    print(f"无法打开摄像头: {camera_index}")
    raise SystemExit(1)

# 可视化参数面板（可选）
panel = DetectorControlPanel(blur_ksize=5, canny_thresh_lower=50, approx_ratio_x100=9, use_roi=1)
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


class DetectorControlPanel:
    """OpenCV 可视化参数调节面板。"""

    def __init__(self, blur_ksize=5, canny_thresh_lower=50, approx_ratio_x100=9, use_roi=1, laser_x=0, laser_y=0, window_name="Controls"):
        self.window_name = window_name
        self.blur_ksize = blur_ksize
        self.canny_thresh_lower = canny_thresh_lower
        self.approx_ratio_x100 = approx_ratio_x100
        self.use_roi = use_roi
        self.laser_x_offset = laser_x
        self.laser_y_offset = laser_y

    @staticmethod
    def _noop(_value):
        return

    def create(self):
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 420, 320)
        cv2.createTrackbar("Blur Ksize", self.window_name, self.blur_ksize, 21, self._noop)
        cv2.createTrackbar("Canny Low", self.window_name, self.canny_thresh_lower, 255, self._noop)
        cv2.createTrackbar("Approx x100", self.window_name, self.approx_ratio_x100, 50, self._noop)
        cv2.createTrackbar("Use ROI", self.window_name, self.use_roi, 1, self._noop)
        cv2.createTrackbar("Laser X", self.window_name, self.laser_x_offset + 100, 200, self._noop)  # 100 is center (+/- 100)
        cv2.createTrackbar("Laser Y", self.window_name, self.laser_y_offset + 100, 200, self._noop)  # 100 is center (+/- 100)

    def sync_to(self, detector):
        current_blur = cv2.getTrackbarPos("Blur Ksize", self.window_name)
        current_canny = cv2.getTrackbarPos("Canny Low", self.window_name)
        current_approx = cv2.getTrackbarPos("Approx x100", self.window_name)
        current_roi = cv2.getTrackbarPos("Use ROI", self.window_name)
        laser_x = cv2.getTrackbarPos("Laser X", self.window_name) - 100
        laser_y = cv2.getTrackbarPos("Laser Y", self.window_name) - 100

        if current_blur < 1:
            current_blur = 1
        if current_blur % 2 == 0:
            current_blur += 1

        detector.blur_ksize = current_blur
        detector.canny_thresh_lower = current_canny
        detector.approx_ratio = max(0.01, current_approx / 100.0)
        detector.use_roi = bool(current_roi)
        detector.laser_offset_x = laser_x
        detector.laser_offset_y = laser_y

class BlankDetector:
    def __init__(self, use_roi=True, blur_ksize=5, canny_thresh_lower=50, approx_ratio=0.09):
        self.use_roi = use_roi
        self.blur_ksize = blur_ksize
        self.canny_thresh_lower = canny_thresh_lower
        self.approx_ratio = approx_ratio
        self.laser_offset_x = 0
        self.laser_offset_y = 0
        
        # Tracking history states
        self.last_bounding_rect = None
        self.roi_lost_frames = 0
        self.area_history = []
        self.center_history = []
        self.target_tracked_frames = 0
        self.TRUST_FRAMES = 5

    def process_frame(self, frame, debug=False):
        """
        处理单帧图像。
        返回: (dx, dy), display_img
        如果未检测到且无法预测，(dx, dy) 返回 None
        """
        if frame is None:
            return None, None
            
        display_img = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        ksize = self.blur_ksize
        if ksize % 2 == 0:
            ksize += 1
        blur = cv2.medianBlur(gray, ksize)
        
        def search_quad(use_roi_flag=False):
            edges_result = cv2.Canny(blur, self.canny_thresh_lower, 200)
            roi_box = None
            
            if use_roi_flag and self.last_bounding_rect is not None:
                x, y, w, h = self.last_bounding_rect
                margin_w = max(30, int(w * 0.3))
                margin_h = max(30, int(h * 0.3))
                
                x1, y1 = max(0, x - margin_w), max(0, y - margin_h)
                x2, y2 = min(blur.shape[1], x + w + margin_w), min(blur.shape[0], y + h + margin_h)
                roi_box = (x1, y1, x2, y2)
                
                relaxed_thresh = max(10, self.canny_thresh_lower - 30)
                roi_img = blur[y1:y2, x1:x2]
                if roi_img.size == 0 or x2 <= x1 or y2 <= y1:
                    roi_box = None
                else:
                    edges_roi = cv2.Canny(roi_img, relaxed_thresh, 200)
                    
                    mask_roi = np.zeros_like(edges_result)
                    cv2.rectangle(mask_roi, (x1, y1), (x2, y2), 255, -1)
                    
                    edges_result[y1:y2, x1:x2] = edges_roi
                    edges_result = cv2.bitwise_and(edges_result, mask_roi)
                
            contours, _ = cv2.findContours(edges_result, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            max_perimeter_search = 0
            best_vert = None
            
            for cnt in contours:
                approx = cv2.approxPolyDP(cnt, self.approx_ratio * cv2.arcLength(cnt, True), True)
                
                if len(approx) == 4:
                    length = cv2.arcLength(approx, True)
                    
                    cosines = []
                    for i in range(4):
                        p0 = approx[i][0]
                        p1 = approx[(i + 1) % 4][0]
                        p2 = approx[(i + 2) % 4][0]
                        v1 = p0 - p1
                        v2 = p2 - p1
                        norm1 = np.linalg.norm(v1)
                        norm2 = np.linalg.norm(v2)
                        if norm1 == 0 or norm2 == 0:
                            angle = 0
                        else:
                            cosine_angle = np.clip(np.dot(v1, v2) / (norm1 * norm2), -1.0, 1.0)
                            angle = np.arccos(cosine_angle) * 180 / np.pi
                        cosines.append(angle)
                    
                    if all(angle >= 30 for angle in cosines):
                        rect = cv2.minAreaRect(approx)
                        rw, rh = rect[1]
                        if rw > 0 and rh > 0:
                            area = cv2.contourArea(approx)
                            box_area = rw * rh
                            rectangularity = area / box_area if box_area > 0 else 0
                            
                            aspect_ratio = max(rw, rh) / min(rw, rh)
                            
                            area_valid = True
                            if len(self.area_history) >= 5:
                                hist = np.array(self.area_history)
                                mean_area = np.mean(hist)
                                std_area = np.std(hist)
                                vels = np.diff(hist)
                                mean_v = np.mean(vels) if len(vels) > 0 else 0
                                pred_area = hist[-1] + mean_v
                                tolerance = max(0.3 * pred_area, 3 * std_area)
                                if not (pred_area - tolerance < area < pred_area + tolerance):
                                    area_valid = False
                            
                            if rectangularity > 0.75 and 1.2 < aspect_ratio < 1.7 and area_valid:
                                if length > max_perimeter_search and length > 10:
                                    max_perimeter_search = length
                                    best_vert = approx
                                    
            return best_vert, edges_result, roi_box

        used_roi = self.use_roi and self.last_bounding_rect is not None
        best_vertices, edges, roi_box_used = search_quad(use_roi_flag=used_roi)
        
        rect_center_current = None
        is_predicted = False
        
        if best_vertices is None and used_roi:
            self.roi_lost_frames += 1
            if self.roi_lost_frames >= 10:
                self.last_bounding_rect = None
                roi_box_used = None
                self.area_history = []
                self.center_history = []
                self.target_tracked_frames = 0
                best_vertices, edges, roi_box_used = search_quad(use_roi_flag=False)
            else:
                if len(self.center_history) >= 2:
                    is_predicted = True
                    hist_arr = np.array(self.center_history)
                    diffs = np.diff(hist_arr, axis=0)
                    mean_v = np.mean(diffs, axis=0)
                    last_c = hist_arr[-1]
                    
                    pred_cx = int(last_c[0] + mean_v[0] * self.roi_lost_frames)
                    pred_cy = int(last_c[1] + mean_v[1] * self.roi_lost_frames)
                    rect_center_current = (pred_cx, pred_cy)
                    
                    if self.last_bounding_rect:
                        _x, _y, _w, _h = self.last_bounding_rect
                        new_x = int(_x + mean_v[0])
                        new_y = int(_y + mean_v[1])
                        self.last_bounding_rect = (new_x, new_y, _w, _h)
                        
                        margin_w, margin_h = max(30, int(_w * 0.3)), max(30, int(_h * 0.3))
                        roi_box_used = (max(0, new_x - margin_w), max(0, new_y - margin_h),
                                        min(blur.shape[1], new_x + _w + margin_w),
                                        min(blur.shape[0], new_y + _h + margin_h))
        
        if best_vertices is not None:
            self.roi_lost_frames = 0
            self.target_tracked_frames += 1
            
            cx = int(np.mean(best_vertices[:, 0, 0]))
            cy = int(np.mean(best_vertices[:, 0, 1]))
            rect_center_current = (cx, cy)
            
            if self.target_tracked_frames >= self.TRUST_FRAMES:
                self.last_bounding_rect = cv2.boundingRect(best_vertices)
                self.area_history.append(cv2.contourArea(best_vertices))
                if len(self.area_history) > 10:
                    self.area_history.pop(0)
                
                self.center_history.append(rect_center_current)
                if len(self.center_history) > 15:
                    self.center_history.pop(0)
            else:
                self.last_bounding_rect = None
                
        elif not is_predicted and not used_roi:
            self.last_bounding_rect = None
            self.roi_lost_frames = 0
            self.area_history = []
            self.center_history = []
            self.target_tracked_frames = 0

        img_h, img_w = display_img.shape[:2]
        # 加入人工激光偏移量后的新目标中心点
        img_center_x = (img_w // 2) + self.laser_offset_x
        img_center_y = (img_h // 2) + self.laser_offset_y
        
        delta_vector = None

        if debug:
            # 绘制带有偏移的虚拟中心准星（代表激光落点）
            cv2.line(display_img, (img_center_x - 10, img_center_y), (img_center_x + 10, img_center_y), (255, 0, 0), 1)
            cv2.line(display_img, (img_center_x, img_center_y - 10), (img_center_x, img_center_y + 10), (255, 0, 0), 1)
            cv2.circle(display_img, (img_center_x, img_center_y), 4, (255, 0, 0), -1)

            if best_vertices is not None:
                cv2.drawContours(display_img, [best_vertices], 0, (0, 255, 0), 2)
                for pt in best_vertices:
                    cv2.circle(display_img, tuple(pt[0]), 4, (0, 0, 255), -1)

        if rect_center_current is not None:
            rect_cx, rect_cy = rect_center_current
            dx = rect_cx - img_center_x
            dy = rect_cy - img_center_y
            
            # 持续输出预测向量（最多10帧），与 ROI 的维持时间一致，超过10帧后将返回 None 令电机停止
            if self.target_tracked_frames >= self.TRUST_FRAMES:
                delta_vector = (dx, dy)
            else:
                delta_vector = None
            
            if debug:
                if self.target_tracked_frames < self.TRUST_FRAMES:
                    # Still verifying trust
                    cv2.circle(display_img, (rect_cx, rect_cy), 4, (0, 165, 255), -1)
                    cv2.putText(display_img, f"Verifying ({self.target_tracked_frames}/{self.TRUST_FRAMES})", 
                                (rect_cx + 8, rect_cy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 1)
                else:
                    color_dot = (0, 255, 255) if not is_predicted else (150, 150, 150)
                    color_line = (255, 0, 255) if not is_predicted else (150, 150, 150)
                    color_text = (0, 0, 255) if not is_predicted else (150, 150, 150)
                    
                    cv2.circle(display_img, (rect_cx, rect_cy), 4, color_dot, -1)
                    cv2.arrowedLine(display_img, (img_center_x, img_center_y), (rect_cx, rect_cy), color_line, 2, tipLength=0.08)
                    
                    prefix = "Vector" if not is_predicted else "Pred Vector"
                    cv2.putText(display_img, f"{prefix} ({dx} px, {dy} px)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_text, 1)
                    
                    if is_predicted:
                        cv2.putText(display_img, "PREDICTING...", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)

        if debug and roi_box_used:
            rx1, ry1, rx2, ry2 = roi_box_used
            cv2.rectangle(display_img, (rx1, ry1), (rx2, ry2), (0, 255, 255), 1)
            cv2.putText(display_img, "ROI", (rx1, ry1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

        return delta_vector, display_img


def run_camera(camera_index=2, width=352, height=288, fps=60):
    """
    独立运行模块进行测试的函数模块，支持传入多个摄像头索引进行多摄选择测试
    """
    detector = BlankDetector()
    cap = open_configured_camera(camera_index, width, height, fps, print_info=False)

    if cap is None:
        print(f"无法打开摄像头: {camera_index}")
        return

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"摄像头 {camera_index} 已开启，分辨率={actual_w}x{actual_h}，fps={actual_fps}；按 'q' 退出")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("读取摄像头帧失败")
            break
            
        h, w = frame.shape[:2]
        if max(h, w) > 1000:
            ratio = 1000.0 / max(h, w)
            frame = cv2.resize(frame, (int(w * ratio), int(h * ratio)))
            
        vector, res_img = detector.process_frame(frame, debug=True)
        
        cv2.imshow("Blank Detector module", res_img)
        
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