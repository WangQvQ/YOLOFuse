from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("ultralytics/cfg/models/fuse/easy-fuse.yaml")
    model.train(
        data="ultralytics/cfg/datasets/LLVIP.yaml",
        ch=6, # 多模态时设置为 6 ，单模态时设置为 3
        imgsz=640,
        epochs=100,
        batch=64,
        close_mosaic=0,
        workers=16,
        device="0",
        optimizer="SGD",
        patience=0,
        amp=False,
        cache=True, # disk 硬盘，速度稍快精度可复现；ram/True 内存，速度快但精度不可复现
        project="runs/train",
        name="easy-fuse",
        resume=False,
        fraction=1, # 只用全部数据的 ？% 进行训练 (0.1-1)
    )
