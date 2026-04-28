import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

class TapeDetectorGUI:
    def __init__(self, window):
        self.window = window
        self.window.title("胶带边界检测调试工具")
        
        # 保存原图，用于实时重新处理
        self.original_img = None
        
        # 摄像头参数
        self.cap = None
        self.is_camera_running = False
        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)

        # --- 界面布局 ---
        self.control_frame = tk.Frame(window)
        self.control_frame.pack(fill=tk.X, pady=5)
        
        self.btn_load = tk.Button(self.control_frame, text="选择图片", command=self.load_image, height=2, width=10)
        self.btn_load.pack(side=tk.LEFT, padx=5)

        self.btn_cam = tk.Button(self.control_frame, text="打开摄像头", command=self.toggle_camera, height=2, width=10)
        self.btn_cam.pack(side=tk.LEFT, padx=5)
        
        # 加入 ROI 跟踪选项 和保存状态
        self.use_roi_var = tk.BooleanVar(value=True)
        self.roi_chk = tk.Checkbutton(self.control_frame, text="启用 ROI 追踪", variable=self.use_roi_var, command=self.process_image)
        self.roi_chk.pack(side=tk.LEFT, padx=5)
        self.last_bounding_rect = None  # 记录上一帧物体的外接矩形 (x, y, w, h)
        self.roi_lost_frames = 0        # 记录 ROI 丢失的连续帧数
        self.area_history = []          # 记录最近 10 帧检测到的对象面积
        self.center_history = []        # 记录最近 15 帧的中心点用于向量预测

        # 边缘检测阈值 & 多边形拟合参数
        # 添加中值滤波的核大小滑动条
        self.blur_k_var = tk.IntVar(value=5)
        self.blur_scale = tk.Scale(self.control_frame, from_=3, to=21, resolution=2, orient=tk.HORIZONTAL,
                                label="中值滤波 Kernel", variable=self.blur_k_var, 
                                length=150, command=self.process_image)
        self.blur_scale.pack(side=tk.LEFT, padx=10)

        self.canny_t_var = tk.IntVar(value=50)
        self.canny_scale = tk.Scale(self.control_frame, from_=10, to=255, orient=tk.HORIZONTAL,
                                label="Canny 检测下限阈值", variable=self.canny_t_var, 
                                length=150, command=self.process_image)
        self.canny_scale.pack(side=tk.LEFT, padx=10)

        self.approx_var = tk.IntVar(value=9)
        self.approx_scale = tk.Scale(self.control_frame, from_=1, to=20, orient=tk.HORIZONTAL,
                                label="多边形逼近参数 (x 0.01)", variable=self.approx_var, 
                                length=150, command=self.process_image)
        self.approx_scale.pack(side=tk.LEFT, padx=10)

        # 创建一个主框架作为滚动区域的容器
        self.main_frame = tk.Frame(window)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # 创建 Canvas 和 Scrollbar
        self.canvas = tk.Canvas(self.main_frame)
        self.v_scrollbar = tk.Scrollbar(self.main_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        self.h_scrollbar = tk.Scrollbar(self.main_frame, orient=tk.HORIZONTAL, command=self.canvas.xview)
        
        self.canvas.configure(yscrollcommand=self.v_scrollbar.set, xscrollcommand=self.h_scrollbar.set)

        # 布局 Scrollbar 和 Canvas
        self.v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # 用于显示图片的容器 (放在 Canvas 内部)
        self.canvas_frame = tk.Frame(self.canvas)
        self.canvas.create_window((0, 0), window=self.canvas_frame, anchor="nw")

        # 当容器尺寸发生变化时，更新 canvas 的滚动区域
        self.canvas_frame.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))

        # 绑定鼠标滚轮事件
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)

        self.panels = []
        labels = ["1. 最终检测(周长最大四边形)", "2. 灰度+模糊滤波", "3. Canny边缘检测", "4. 多边形逼近验证"]
        for i in range(4):
            frame = tk.Frame(self.canvas_frame)
            frame.grid(row=i // 2, column=i % 2, padx=5, pady=5)
            tk.Label(frame, text=labels[i] + " (双击查看大图)").pack()
            panel = tk.Label(frame)
            panel.pack()
            # 绑定双击事件，传递索引 i
            panel.bind("<Double-Button-1>", lambda e, idx=i: self.show_large_image(idx))
            self.panels.append(panel)

        self.current_results = [] # 保存当前处理的4张高清大图

    def show_large_image(self, idx):
        if not self.current_results or idx >= len(self.current_results):
            return
            
        img = self.current_results[idx]
        
        top = tk.Toplevel(self.window)
        top.title(f"大图预览 - 面板 {idx+1}")
        
        # 适当缩放以适应屏幕，防图片太大超屏
        h, w = img.shape[:2]
        max_size = 1000
        if max(h, w) > max_size:
            ratio = max_size / max(h, w)
            img = cv2.resize(img, (int(w * ratio), int(h * ratio)))
            
        if len(img.shape) == 2:
            rgb_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
        img_pil = Image.fromarray(rgb_img)
        img_tk = ImageTk.PhotoImage(image=img_pil)
        
        lbl = tk.Label(top, image=img_tk)
        lbl.pack()
        lbl.image = img_tk  # 防止被回收

    def _on_mousewheel(self, event):
        self.canvas.yview_scroll(int(-1*(event.delta/120)), "units")

    def toggle_camera(self):
        if not self.is_camera_running:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                print("无法打开摄像头，请检查连接或权限")
                return
            self.is_camera_running = True
            self.btn_cam.config(text="关闭摄像头")
            self.btn_load.config(state=tk.DISABLED) # 开启摄相机时禁用图片选择
            self.video_loop()
        else:
            self.is_camera_running = False
            self.btn_cam.config(text="打开摄像头")
            self.btn_load.config(state=tk.NORMAL)
            if self.cap:
                self.cap.release()
                self.cap = None

    def video_loop(self):
        if self.is_camera_running and self.cap:
            ret, frame = self.cap.read()
            if ret:
                h, w = frame.shape[:2]
                if max(h, w) > 1000:
                    ratio = 1000.0 / max(h, w)
                    frame = cv2.resize(frame, (int(w * ratio), int(h * ratio)))
                
                self.original_img = frame
                self.process_image()
            
            # 使用 after 方法不断刷新
            self.window.after(30, self.video_loop)

    def on_closing(self):
        self.is_camera_running = False
        if self.cap:
            self.cap.release()
        self.window.destroy()

    def load_image(self):
        path = filedialog.askopenfilename()
        if not path: return
        
        # 1. 读取原图并保存，以供滑动条拖动时频繁调用
        img = cv2.imread(path)
        if img is None: return
        
        # 将原图下采样到最大边长为 1000px，统一不同分别率图片的特征尺度，并提速
        h, w = img.shape[:2]
        if max(h, w) > 1000:
            ratio = 1000.0 / max(h, w)
            img = cv2.resize(img, (int(w * ratio), int(h * ratio)))
            
        self.original_img = img
        self.process_image()

    def process_image(self, *args):
        if self.original_img is None: return
        
        display_img = self.original_img.copy()
        debug_img = np.zeros_like(display_img)
        
        # 1. 灰度化与模糊去噪
        gray = cv2.cvtColor(self.original_img, cv2.COLOR_BGR2GRAY)
        
        # 获取滑动条设定的中值滤波核大小 (必须是奇数)
        ksize = self.blur_k_var.get()
        if ksize % 2 == 0: ksize += 1
        blur = cv2.medianBlur(gray, ksize)
        
        # 2. Canny 边缘检测
        canny_thresh_lower = self.canny_t_var.get()
        approx_ratio = self.approx_var.get() / 100.0

        def search_quad(use_roi=False):
            # Canny 边缘检测
            edges_result = cv2.Canny(blur, canny_thresh_lower, 200)
            
            roi_box = None
            if use_roi and self.last_bounding_rect is not None:
                # 提取上一帧矩形，并根据其大小动态缩放扩张 (如各边向外延伸 30%)
                x, y, w, h = self.last_bounding_rect
                
                # 动态计算 margin：依据检测框尺寸扩展 30%，同时给出最小 30 像素的下限兜底防小框过严
                margin_w = max(30, int(w * 0.3))
                margin_h = max(30, int(h * 0.3))
                
                # 约束在图像范围内
                x1, y1 = max(0, x - margin_w), max(0, y - margin_h)
                x2, y2 = min(blur.shape[1], x + w + margin_w), min(blur.shape[0], y + h + margin_h)
                roi_box = (x1, y1, x2, y2)
                
                # “注意力”放宽参数机制：在 ROI 内使用更低的 Canny 下限阈值来捕捉边缘
                relaxed_thresh = max(10, canny_thresh_lower - 30)
                edges_roi = cv2.Canny(blur[y1:y2, x1:x2], relaxed_thresh, 200)
                
                # 注意力聚焦掩码：屏蔽 ROI 区域外的所有边缘干扰
                mask_roi = np.zeros_like(edges_result)
                cv2.rectangle(mask_roi, (x1, y1), (x2, y2), 255, -1)
                
                # 将放宽检测的 ROI 内边缘，覆盖掉原图边缘
                edges_result[y1:y2, x1:x2] = edges_roi
                edges_result = cv2.bitwise_and(edges_result, mask_roi)
                
            # 3. 寻找最大的合法四边形，周长的最大值筛选
            contours, _ = cv2.findContours(edges_result, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(debug_img, contours, -1, (0, 100, 100), 1)
            
            max_perimeter_search = 0
            best_vert = None
            
            for cnt in contours:
                approx = cv2.approxPolyDP(cnt, approx_ratio * cv2.arcLength(cnt, True), True)
                
                # 此处判断非常严格：逼近后多边形的顶点如果不是4，视为无效
                if len(approx) == 4:
                    length = cv2.arcLength(approx, True)
                    cv2.drawContours(debug_img, [approx], 0, (255, 0, 0), 2)
                    
                    # 开始内角约束判断 (仓库要求最小内角 > 30 度)
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
                            cosine_angle = np.dot(v1, v2) / (norm1 * norm2)
                            cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
                            angle = np.arccos(cosine_angle) * 180 / np.pi
                        cosines.append(angle)
                    
                    # 周长需要大于目前最大值且满足最小内角30度的限制
                    if all(angle >= 30 for angle in cosines):
                        # 计算面积与最小外接矩形，以评估矩形度和长宽比
                        rect = cv2.minAreaRect(approx)
                        rw, rh = rect[1]
                        if rw > 0 and rh > 0:
                            # 1. 矩形度检测：自身面积 vs 最小外接矩形面积，理论长方形的值无限趋近于1
                            area = cv2.contourArea(approx)
                            box_area = rw * rh
                            rectangularity = area / box_area if box_area > 0 else 0
                            
                            # 2. 长宽比检测：大边 / 小边，A4 尺寸约为 297/210 ≈ 1.414，这里给适当放宽 1.2 ~ 1.7 的包容度(考虑透视畸变)
                            aspect_ratio = max(rw, rh) / min(rw, rh)
                            
                            # 3. 统计学面积过滤魔法 (如果有历史记录的话)
                            area_valid = True
                            if len(self.area_history) >= 5:
                                hist = np.array(self.area_history)
                                mean_area = np.mean(hist)
                                std_area = np.std(hist)
                                
                                # 根据最近帧计算变化速率 (斜率)
                                vels = np.diff(hist)
                                mean_v = np.mean(vels) if len(vels) > 0 else 0
                                
                                # 预测下一帧面积，考虑到一定的加缩放
                                pred_area = hist[-1] + mean_v
                                
                                # 动态窗口：容许基础容差点(预测面积的30%) 与 波动率(三倍标注差)取大者
                                tolerance = max(0.3 * pred_area, 3 * std_area)
                                
                                if not (pred_area - tolerance < area < pred_area + tolerance):
                                    area_valid = False
                            
                            if rectangularity > 0.75 and 1.2 < aspect_ratio < 1.7 and area_valid:
                                if length > max_perimeter_search and length > 10:
                                    max_perimeter_search = length
                                    best_vert = approx
                            
            return best_vert, edges_result, roi_box

        # ==== 开启检测逻辑 ====
        # 尝试 ROI 重复检查
        used_roi = self.use_roi_var.get() and self.last_bounding_rect is not None
        best_vertices, edges, roi_box_used = search_quad(use_roi=used_roi)
        
        rect_center_current = None
        is_predicted = False
        
        # 丢帧/降级机制：冻结 ROI 几帧
        if best_vertices is None and used_roi:
            self.roi_lost_frames += 1
            if self.roi_lost_frames >= 15:  # 容忍丢框15帧 (约半秒) 再去全画幅找
                self.last_bounding_rect = None
                roi_box_used = None
                self.area_history = []
                self.center_history = []
                best_vertices, edges, roi_box_used = search_quad(use_roi=False)
            else:
                # 就算丢了目标，我也要预测出它滑向了哪里！根据过去15帧历史速度进行预测
                if len(self.center_history) >= 2:
                    is_predicted = True
                    hist_arr = np.array(self.center_history)
                    diffs = np.diff(hist_arr, axis=0) # 计算两两帧之间的位移向量
                    mean_v = np.mean(diffs, axis=0)   # 过去N帧的平均运动速度向量
                    last_c = hist_arr[-1]
                    
                    # 预测目前应该出现的中心点
                    pred_cx = int(last_c[0] + mean_v[0] * self.roi_lost_frames)
                    pred_cy = int(last_c[1] + mean_v[1] * self.roi_lost_frames)
                    rect_center_current = (pred_cx, pred_cy)
                    
                    # 使 ROI 注意力框也跟着预测速度发生位移，防止因运动太快跑出冻结的原框！
                    if self.last_bounding_rect:
                        _x, _y, _w, _h = self.last_bounding_rect
                        new_x = int(_x + mean_v[0])
                        new_y = int(_y + mean_v[1])
                        self.last_bounding_rect = (new_x, new_y, _w, _h) # 更新记忆，让 ROI 运动起来
                        
                        # 更新当前在第一块面板上高亮的黄虚线显示框，让用户肉眼可见它的位置预测
                        margin_w, margin_h = max(30, int(_w * 0.3)), max(30, int(_h * 0.3))
                        roi_box_used = (max(0, new_x - margin_w), max(0, new_y - margin_h),
                                        min(blur.shape[1], new_x + _w + margin_w),
                                        min(blur.shape[0], new_y + _h + margin_h))
        
        # 无论此时经过上面的一系列挽救后结果如何，处理记忆组装
        if best_vertices is not None:
            self.roi_lost_frames = 0
            self.last_bounding_rect = cv2.boundingRect(best_vertices)
            self.area_history.append(cv2.contourArea(best_vertices))
            if len(self.area_history) > 10:
                self.area_history.pop(0)
                
            # 加入中心坐标记录用于下文预测
            cx = int(np.mean(best_vertices[:, 0, 0]))
            cy = int(np.mean(best_vertices[:, 0, 1]))
            rect_center_current = (cx, cy)
            self.center_history.append(rect_center_current)
            if len(self.center_history) > 15:
                self.center_history.pop(0)
                
        elif not is_predicted: 
            # 既没有真实找到，又不能被预测，这说明真·丢失了
            self.last_bounding_rect = None
            self.roi_lost_frames = 0
            self.area_history = []
            self.center_history = []

        # 绘制中心点与目标框
        img_h, img_w = display_img.shape[:2]
        img_center_x, img_center_y = img_w // 2, img_h // 2
        cv2.circle(display_img, (img_center_x, img_center_y), 8, (255, 0, 0), -1) # 图像中心点 (蓝色)

        # 绘制确认检测出的最佳四边形 (仅检测成功时绘制)
        if best_vertices is not None:
            cv2.drawContours(display_img, [best_vertices], 0, (0, 255, 0), 6)
            for pt in best_vertices:
                cv2.circle(display_img, tuple(pt[0]), 12, (0, 0, 255), -1)
                
        # 绘制真实或预测的目标中心、追踪向量箭头及文字
        if rect_center_current is not None:
            rect_cx, rect_cy = rect_center_current
            
            # 如果是处于瞎掉期间的“预测态”，我们改用灰色和不同的标记方式给个提示
            color_dot = (0, 255, 255) if not is_predicted else (150, 150, 150)
            color_line = (255, 0, 255) if not is_predicted else (150, 150, 150)
            color_text = (0, 0, 255) if not is_predicted else (150, 150, 150)
            
            cv2.circle(display_img, (rect_cx, rect_cy), 8, color_dot, -1)
            cv2.arrowedLine(display_img, (img_center_x, img_center_y), (rect_cx, rect_cy), color_line, 6, tipLength=0.08)
            
            dx = rect_cx - img_center_x
            dy = rect_cy - img_center_y
            prefix = "Vector" if not is_predicted else "Pred Vector"
            cv2.putText(display_img, f"{prefix}: ({dx} px, {dy} px)", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 2.0, color_text, 4)
            
            if is_predicted:
                cv2.putText(display_img, "PREDICTING...", (50, 140), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 165, 255), 4)

        # 如使用注意力的话可视化 ROI 边框
        if roi_box_used:
            rx1, ry1, rx2, ry2 = roi_box_used
            # 绘制 ROI 黄色虚线框
            cv2.rectangle(display_img, (rx1, ry1), (rx2, ry2), (0, 255, 255), 2)
            cv2.putText(display_img, "ROI Attention", (rx1, ry1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        # --- 更新界面四个面板 ---
        results = [display_img, blur, edges, debug_img]
        self.current_results = results # 保存高清供大图显示
        
        for i in range(4):
            self.update_panel(self.panels[i], results[i])

    def update_panel(self, panel, cv_img):
        # 统一缩放尺寸便于显示
        h, w = cv_img.shape[:2]
        aspect = w / h
        new_w = 400
        new_h = int(new_w / aspect)
        
        resized = cv2.resize(cv_img, (new_w, new_h))
        
        # 转换颜色空间以适配 Tkinter
        if len(resized.shape) == 2: # 灰度图
            rgb_img = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
        else: # BGR 转换成 RGB
            rgb_img = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            
        img_pil = Image.fromarray(rgb_img)
        img_tk = ImageTk.PhotoImage(image=img_pil)
        
        panel.configure(image=img_tk)
        panel.image = img_tk

# 启动程序
if __name__ == "__main__":
    root = tk.Tk()
    app = TapeDetectorGUI(root)
    root.mainloop()