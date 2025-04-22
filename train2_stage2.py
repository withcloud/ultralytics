from ultralytics import YOLO

# Load the model from the first training
model = YOLO("runs/pose/train41/weights/last.pt")  # 使用第一次訓練的最終權重

# Train the model with fine-tuning parameters
results = model.train(
    data="coco-pose.yaml",
    epochs=100,
    imgsz=640,
    batch=192,
    save_period=1,
    cache="disk",
    optimizer="AdamW",
    lr0=0.0005,  # 降低學習率
    lrf=0.01,
    cos_lr=True,  # 使用餘弦學習率調度
    close_mosaic=15,  # 提前關閉馬賽克增強
    device="0,1",
    patience=50  # 提前停止條件
)
