from ultralytics import YOLO

model = YOLO("yolo11m-pose.pt")

metrics = model.val(
    data="coco-pose.yaml",
)
