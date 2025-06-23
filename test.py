from ultralytics import YOLO

if __name__ == '__main__':
    # Load a model
    model = YOLO('ultralytics/cfg/models/fuse/后期融合.yaml')  # build a new model from YAML
    model.info()

