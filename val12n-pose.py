from ultralytics import YOLO

model = YOLO("yolo12n-pose.pt")

metrics = model.val(
    data="coco-pose.yaml",
)
