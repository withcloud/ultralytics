from ultralytics import YOLO
model = YOLO("yolo11n-pose")
model.export(format="tfjs")
