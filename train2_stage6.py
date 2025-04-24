from ultralytics import YOLO

# Load the model from the fifth training stage best weights
model = YOLO("/root/autodl-tmp/ultralytics/runs/pose/train52/weights/best.pt")  # 使用第五次訓練的最佳權重

# Train the model with focus on pose keypoint precision - adjusted for 640 validation set
results = model.train(
    data="coco-pose.yaml",
    epochs=150,            # 增加訓練輪數
    imgsz=640,             # 調整為與驗證集一致的大小
    batch=128,             # 增加批次大小，利用降低imgsz釋放的顯存
    save_period=1,         # 每個epoch保存
    cache="disk",          # 使用磁盤緩存
    optimizer="AdamW",     # 繼續使用AdamW優化器
    lr0=0.00002,           # 微調學習率，考慮到較小圖像尺寸
    lrf=0.005,             # 保持較低的最終學習率因子
    cos_lr=True,           # 餘弦學習率調度
    warmup_epochs=3.0,     # 稍微增加熱身階段
    device="0,1",          # 使用兩個GPU
    patience=40,           # 增加耐心值
    box=10.0,              # 框損失權重
    pose=28.0,             # 進一步增加姿態損失權重
    kobj=7.0,              # 進一步增加關鍵點對象損失權重
    close_mosaic=10,       # 延後關閉馬賽克，增強數據多樣性
    amp=True,              # 啟用混合精度訓練
    overlap_mask=True,     # 重疊口罩
    multi_scale=True,      # 啟用多尺度訓練，提高模型對不同尺寸的適應性
    hsv_h=0.015,           # 保持減少的色調增強
    hsv_s=0.2,             # 保持減少的飽和度增強
    hsv_v=0.2,             # 保持減少的亮度增強
    copy_paste=0.0,        # 保持關閉複製粘貼
    mixup=0.0,             # 保持關閉mixup
    degrees=0.0,           # 保持關閉旋轉增強
    translate=0.1,         # 保持減少的平移增強
    scale=0.1,             # 保持減少的縮放增強
    fliplr=0.5,            # 保持水平翻轉
    nbs=64,                # 指定標稱批次大小
    val=True,              # 確保每個epoch都進行驗證
    rect=False,            # 對於姿態估計，保持方形輸入
    resume=False           # 從頭開始新的訓練
) 