from ultralytics import YOLO

model = YOLO("yolov12n.pt")
model.val(
    data="coco128.yaml", # VOC2007.yaml / pcb.yaml
    save_json=False)  
