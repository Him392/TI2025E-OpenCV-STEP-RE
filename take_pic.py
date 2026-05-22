import cv2
import os
import time
from datetime import datetime

def main():
    # 1. 设置保存路径并确保目录存在
    save_dir = r"D:/yolo/takepic"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 2. 初始化摄像头并配置参数
    cap = cv2.VideoCapture(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 252)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 288)

    if not cap.isOpened():
        print("无法打开摄像头，请检查摄像头索引是否正确。")
        return

    print(f"开始抓拍，照片将保存至: {save_dir}")
    print("按 'q' 或 'ESC' 键退出程序。")
    print("按 '空格' 键暂停/继续抓拍。")

    pic_index = 1
    last_capture_time = time.time()
    is_paused = True

    while True:
        # 持续读取画面刷新缓存，保证拍到的不是滞后的旧画面
        ret, frame = cap.read()
        if not ret:
            print("无法获取画面，退出...")
            break

        # 拷贝一份用于显示GUI，保持抓拍的 frame 原始纯净
        display_frame = frame.copy()

        # 在画面上叠加当前状态
        if is_paused:
            cv2.putText(display_frame, "PAUSED", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            # 暂停时不断更新时间，确保恢复后重新计时1.5秒
            last_capture_time = time.time()
        else:
            cv2.putText(display_frame, "RECORDING", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # 显示实时预览 (如果不需要预览可以注释掉这两句)
        cv2.imshow("Auto Capture", display_frame)

        # 3. 每隔 1.5 秒抓拍并保存一张纯净的图片
        current_time = time.time()
        if not is_paused and current_time - last_capture_time >= 1.5:
            # 格式化当前时间为 YYYYMMDD-HHMMSS
            time_str = datetime.now().strftime("%Y%m%d-%H%M%S")
            # 拼接完整文件名如：20250521-111701-1.jpg
            file_name = f"{time_str}-{pic_index}.jpg"
            file_path = os.path.join(save_dir, file_name)
            
            # 保存图片
            cv2.imwrite(file_path, frame)
            print(f"已保存照片: {file_name}")
            
            pic_index += 1
            last_capture_time = current_time

        # 1ms等待时间检测键盘打断信号
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key == ord(' '):  # 按空格切换暂停/恢复状态
            is_paused = not is_paused
            if is_paused:
                print(">>> 抓拍已暂停 >>>")
            else:
                print("<<< 抓拍已恢复 <<<")

    # 释放资源
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()