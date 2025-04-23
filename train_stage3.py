from ultralytics import YOLO

# Load the model from the second training stage
model = YOLO("/root/autodl-tmp/ultralytics/runs/pose/train43/weights/last.pt")  # 使用第二次訓練的最佳權重

# Train the model with advanced fine-tuning parameters
results = model.train(
    data="coco-pose.yaml",
    epochs=50,
    imgsz=1280,  # 提高圖像解析度
    batch=32,    # 降低批次大小以適應更大解析度
    save_period=1,
    cache="disk",
    optimizer="AdamW",
    lr0=0.0001,  # 進一步降低學習率
    lrf=0.01,
    cos_lr=True,
    close_mosaic=0,  # 完全關閉馬賽克增強
    patience=20,
    overlap_mask=True,
    box=10.0,    # 增加框損失權重
    pose=15.0,   # 增加姿態損失權重
    kobj=3.0,    # 增加關鍵點對象損失權重
    warmup_epochs=1.0  # 短暫的熱身期
) 