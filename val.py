from ultralytics import YOLO

# Load a model
model = YOLO("yolo11m-pose.pt")

# Validate the model
metrics = model.val(
  data="yoga82.yaml",
)
