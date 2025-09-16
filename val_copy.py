from ultralytics import YOLO

# 加載更新後的模型
model = YOLO('models/gde-pose-640.pt')

# 使用測試集評估模型性能
metrics = model.val(
  data="coco-pose.yaml"
)

print(f"mAP@50: {metrics.box.map50}")
print(f"Pose Keypoint mAP: {metrics.pose.map}")
