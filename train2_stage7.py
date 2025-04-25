from ultralytics import YOLO

# 加載第六次訓練的最佳權重
model = YOLO("/root/autodl-tmp/ultralytics/runs/pose/train55/weights/best.pt")  # 使用第六次訓練的最佳權重

# 專注於提高pose mAP50-95的訓練策略
results = model.train(
    data="coco-pose.yaml",
    epochs=50,            # 保持epochs數量
    imgsz=1280,           # 降回到1280，與第六次訓練保持一致
    batch=64,             # 減少批次大小以提高精度
    save_period=1,        # 每個epoch保存
    cache="disk",         # 使用磁盤緩存
    optimizer="AdamW",    # 繼續使用AdamW優化器
    lr0=0.000005,         # 使用更低的學習率進行精細微調
    lrf=0.0005,           # 更小的最終學習率因子
    cos_lr=True,          # 餘弦學習率調度
    warmup_epochs=2.0,    # 適度的熱身期
    device="0,1,2,3",     # 使用全部4個GPU
    patience=40,          # 保持耐心值
    box=12.0,             # 略微增加框損失權重
    pose=35.0,            # 大幅增加姿態損失權重
    kobj=10.0,            # 增加關鍵點對象損失權重
    close_mosaic=0,       # 完全關閉馬賽克增強
    amp=True,             # 啟用混合精度訓練
    overlap_mask=True,    # 啟用重疊口罩
    
    # 數據增強參數 (最小化增強以專注於原始數據學習)
    hsv_h=0.005,          # 最小色調變化 (0.0-1.0)
    hsv_s=0.05,           # 最小飽和度變化 (0.0-1.0)
    hsv_v=0.05,           # 最小亮度變化 (0.0-1.0)
    degrees=0.0,          # 關閉旋轉增強 (0.0-180.0)
    translate=0.02,       # 最小平移增強 (0.0-1.0)
    scale=0.02,           # 最小縮放增強 (>=0.0)
    shear=0.0,            # 關閉剪切增強 (-180.0-180.0)
    perspective=0.0,      # 關閉透視變換 (0.0-0.001)
    flipud=0.0,           # 關閉垂直翻轉 (0.0-1.0)
    fliplr=0.5,           # 保留水平翻轉 (0.0-1.0)
    mosaic=0.0,           # 關閉馬賽克增強 (0.0-1.0)
    mixup=0.0,            # 關閉Mixup增強 (0.0-1.0)
    copy_paste=0.0,       # 關閉複製粘貼增強 (0.0-1.0)
) 