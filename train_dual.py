from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("ultralytics/cfg/models/YOLOFuse/特征级融合-中期.yaml")
    model.train(
        data="ultralytics/cfg/datasets/LLVIP.yaml",
        ch=6, # 多模态时设置为 6 ，单模态时设置为 3
        imgsz=640,
        epochs=300,
        batch=64,
        close_mosaic=0,
        workers=16,
        device="0",
        optimizer="SGD",
        patience=0,
        amp=False,
        cache=True,
        project="runs/train",
        name="RGB-IR-特征级融合-中期",
        resume=False,
        fraction=1, # 只用全部数据的 ？% 进行训练 (0.1-1)
    )
