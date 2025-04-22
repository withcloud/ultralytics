from ultralytics import YOLO

model = YOLO("yolo12m-pose.pt")

metrics = model.val(
    data="coco-pose.yaml",
)
