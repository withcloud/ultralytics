from ultralytics import YOLO

model = YOLO("yolo12x-pose.pt")

metrics = model.val(
    data="coco-pose.yaml",
)
