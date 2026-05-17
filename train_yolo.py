from ultralytics import YOLO

def main():
    # 加载预训练模型
    model = YOLO('D:/yolo/yolo11n.pt') 

    # 开始训练
    # 注意：针对 2GB 显存，batch 和 imgsz 参数已经做了保守设置以防 OOM
    results = model.train(
        data='D:/yolo/dataset.yaml',  # 数据集配置文件路径
        epochs=100,                   # 训练轮数
        batch=2,                      # 批次大小 (调小以防显存溢出)
        imgsz=320,                    # 图片尺寸 (调小以防显存溢出)
        workers=2,                    # 数据加载线程数
        device='0',                   # 指定GPU0
        project='runs/train',         # 训练结果保存路径
        name='yolov11_custom'         # 本次实验名称
    )

if __name__ == '__main__':
    # Windows 下多进程需要这行防保护
    main()
