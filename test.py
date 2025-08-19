from ultralytics import YOLO

if __name__ == "__main__":
    # Load a model
    cfg_path = 'ultralytics/cfg/models/fuse/Easy-level-Feature-Fusion.yaml'
    model = YOLO(cfg_path)
    model._new(cfg_path, task='detect', verbose=True)

