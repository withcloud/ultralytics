from ultralytics import YOLO

model = YOLO("yolo11x-pose.pt")

metrics = model.val(
    data="coco-pose.yaml",
)
