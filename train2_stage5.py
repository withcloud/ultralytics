from ultralytics import YOLO

# Load the model from the fourth training stage best weights
model = YOLO("/root/autodl-tmp/ultralytics/runs/pose/train49/weights/best.pt")  # 使用第四次訓練的最佳權重

# Train the model with enhanced data augmentation and fine-tuned parameters
results = model.train(
    data="coco-pose.yaml",
    epochs=100,
    imgsz=1280,        # 提高回1280解析度
    batch=64,          # 增加批次大小以充分利用GPU
    save_period=1,     # 每個epoch保存
    cache="disk",      # 使用磁盤緩存
    optimizer="AdamW", # 繼續使用AdamW優化器
    lr0=0.00002,       # 因增加批次大小而稍微提高學習率
    lrf=0.01,          # 最終學習率因子
    cos_lr=True,       # 餘弦學習率調度
    warmup_epochs=2.0, # 熱身階段
    device="0,1",      # 使用兩個GPU
    patience=30,       # 增加耐心值
    box=15.0,          # 進一步增加框損失權重
    pose=22.0,         # 進一步增加姿態損失權重 
    kobj=5.0,          # 增加關鍵點對象損失權重
    nbs=64,            # 調整標稱批次大小
    degrees=10.0,      # 增加旋轉增強
    translate=0.2,     # 增加平移增強
    scale=0.2,         # 增加縮放增強
    shear=5.0,         # 增加剪切增強
    perspective=0.001, # 增加透視增強
    flipud=0.1,        # 上下翻轉增強
    mosaic=0.5,        # 降低馬賽克增強概率
    mixup=0.15,        # 增加mixup增強
    copy_paste=0.3,    # 增加複製粘貼增強
    auto_augment="randaugment",  # 使用隨機增強
    amp=True,          # 啟用混合精度訓練
    overlap_mask=True  # 重疊口罩
) 
