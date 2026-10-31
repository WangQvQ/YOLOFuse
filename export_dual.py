"""Export an RGB+IR checkpoint with its architecture's six input channels."""

from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("runs/train/easy-fuse/weights/best.pt")
    model.export(format="onnx", imgsz=640, opset=17, simplify=True)
    # For TorchScript: model.export(format="torchscript", imgsz=640)
