from ultralytics import YOLO

def main():
    # 恢复训练：加载中断那次保存的 last.pt
    model = YOLO('D:\yolo\yolo11s.pt') 

    # 开始训练
    # 注意：针对 2GB 显存，batch 和 imgsz 参数已经做了保守设置以防 OOM
    results = model.train(
        data='D:/yolo/dataset.yaml',  # 数据集配置文件路径
        epochs=500,                   # 训练轮数
        batch=8,                      # 批次大小 (调小以防显存溢出)
        imgsz=320,                    # 图片尺寸 (调小以防显存溢出)
        workers=0,                    # 修改为0：解决 Windows 下多线程导致的 I/O 关闭冲突
        device='0',                   # 指定GPU
        project='runs/train',         # 训练结果保存路径
        name='yolov11m_0521_',        # 本次实验名称
    )

if __name__ == '__main__':
    # Windows 下多进程需要这行防保护
    main()
