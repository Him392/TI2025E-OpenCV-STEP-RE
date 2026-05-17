import cv2
from ultralytics import YOLO
import sys
import os

def main():
    # 1. 设置权重路径和视频路径
    weights_path = "best.pt"
    video_path = r"test_video.mp4"  # 替换成您实际要测试的视频路径
    
    if not os.path.exists(weights_path):
        print(f"找不到权重文件：{weights_path}。请检查路径。")
        # 也可以换成同目录下的 best.pt
        weights_path = "best.pt" if os.path.exists("best.pt") else weights_path

    # 2. 加载您训练好的最佳模型权重
    model = YOLO(weights_path)

    # 3. 打开视频 (如果是外接 USB 摄像头，填 0 或 1)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"无法打开视频：{video_path}！请检查视频所在路径。")
        sys.exit()

    # --- 新增功能：设置窗口可缩放 ---
    window_name = "YOLOv11 Detection Demo"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)  # 使得窗口可以通过鼠标拖拽缩放
    
    # --- 新增功能：设置视频进度条 ---
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    def nothing(x):
        pass
    if total_frames > 0:
        cv2.createTrackbar("Progress", window_name, 0, total_frames, nothing)

    print("已成功打开视频开始检测。按 'q' 或 'ESC' 退出，按 '空格' 键暂停/继续。")

    is_paused = False

    while cap.isOpened():
        # --- 进度条检测联动逻辑 ---
        if total_frames > 0:
            trackbar_pos = cv2.getTrackbarPos("Progress", window_name)
            current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            
            # 如果鼠标拖动了进度条(与真实帧数存在落差)，则让视频跳到对应帧
            if abs(trackbar_pos - current_frame) > 2:
                cap.set(cv2.CAP_PROP_POS_FRAMES, trackbar_pos)
                # 暂停状态下拖动进度条，强制读取一帧用于刷新画面
                if is_paused:
                    success, frame = cap.read()
                    if success:
                        # 使用 track 替代 predict，开启 persist=True 来分配追踪 ID (即编号)
                        # 添加 imgsz=320 降低分辨率，改用更快的以提升帧率 bytetrack 追踪器
                        results = model.track(source=frame, conf=0.75, persist=True, verbose=False, imgsz=320, tracker="bytetrack.yaml")
                        annotated_frame = results[0].plot()
                        cv2.imshow(window_name, annotated_frame)
            else:
                # 正常播放时，让进度条跟着视频帧自动跑
                if not is_paused:
                    cv2.setTrackbarPos("Progress", window_name, current_frame)

        if not is_paused:
            success, frame = cap.read()
            if not success:
                print("视频播放完毕或无法读取新帧。")
                break

            # 4. 对当前帧进行检测和追踪，设置 置信度阈值为 0.75
            # 使用 track 并设置 persist=True，YOLO会自动给画面里的物体加上稳定的追踪编号(ID)
            # 添加 imgsz=320 降低分析分辨率，改用 tracker="bytetrack.yaml" 换用更轻量的追踪算法
            results = model.track(source=frame, conf=0.75, persist=True, verbose=False, imgsz=320, tracker="bytetrack.yaml")

            # 5. 获取带有检测框和标签的渲染画面
            annotated_frame = results[0].plot()

            # 6. 显示画面
            cv2.imshow(window_name, annotated_frame)

        # 按下 'q' 或 'Esc' 键退出，空格暂停
        # 等待时间改为 1ms 释放最大性能，不再人为增加延迟
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key == ord(' '):  # 按空格键切换暂停/播放状态
            is_paused = not is_paused

    # 释放资源
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
