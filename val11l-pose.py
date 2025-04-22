from ultralytics import YOLO

model = YOLO("yolo11l-pose.pt")

metrics = model.val(
    data="coco-pose.yaml",
)
