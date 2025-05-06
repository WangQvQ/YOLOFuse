from ultralytics import YOLO

model = YOLO("runs/train/RGB-IR/weights/best.pt")
model.val(
    data="ultralytics/cfg/datasets/LLVIP.yaml",
    ch=6,  # 多模态时设置为 6 ，单模态时设置为 3
)
