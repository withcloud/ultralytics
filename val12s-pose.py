from ultralytics import YOLO

model = YOLO("yolo12s-pose.pt")

metrics = model.val(
    data="coco-pose.yaml",
)
