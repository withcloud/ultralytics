from ultralytics import YOLO

model = YOLO("yolo12l-pose.pt")

metrics = model.val(
    data="coco-pose.yaml",
)
