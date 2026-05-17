from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("runs/train/easy-fuse/weights/best.pt")
    model.val(
        data="ultralytics/cfg/datasets/LLVIP.yaml",
        ch=6,  # 多模态时设置为 6 ，单模态时设置为 3
    )
