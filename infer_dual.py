from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("runs/train/RGB-IR/weights/best.pt")

    model.predict(
        source="ultralytics/assets/LLVIP/images",
        batch=1,
        save=True,
    )
