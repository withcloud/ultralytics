from ultralytics import YOLO

# Load the model from the fifth training stage best weights
model = YOLO("/root/autodl-tmp/ultralytics/runs/pose/train55/weights/last.pt")  # 使用第五次訓練的最佳權重

# Train the model with focus on pose keypoint precision
results = model.train(
    data="coco-pose.yaml",
    epochs=100,
    imgsz=1280,          # 維持高解析度
    batch=64,            # 保持批次大小
    save_period=1,       # 每個epoch保存
    cache="disk",        # 使用磁盤緩存
    optimizer="AdamW",   # 繼續使用AdamW優化器
    lr0=0.000015,        # 再次降低學習率以實現精細微調
    lrf=0.005,           # 更低的最終學習率因子，更精細調整
    cos_lr=True,         # 餘弦學習率調度
    warmup_epochs=2.0,   # 適度的熱身階段
    device="0,1",        # 使用兩個GPU
    patience=35,         # 增加耐心值，避免過早停止
    box=12.0,            # 降低框損失權重，更專注於姿態
    pose=25.0,           # 進一步增加姿態損失權重
    kobj=6.0,            # 增加關鍵點對象損失權重
    close_mosaic=5,      # 提前關閉馬賽克，讓模型在真實情況下訓練更久
    amp=True,            # 啟用混合精度訓練
    overlap_mask=True,   # 重疊口罩
    augment=True,        # 增強數據增強
    hsv_h=0.015,         # 減少色調增強
    hsv_s=0.2,           # 減少飽和度增強
    hsv_v=0.2,           # 減少亮度增強
    copy_paste=0.0,      # 關閉複製粘貼
    mixup=0.0,           # 關閉mixup
    degrees=0.0,         # 關閉旋轉增強
    translate=0.1,       # 減少平移增強
    scale=0.1,           # 減少縮放增強
    fliplr=0.5,          # 保留水平翻轉
    resume=True
) 