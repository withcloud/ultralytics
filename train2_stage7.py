from ultralytics import YOLO

# 加載第六次訓練的最佳權重
model = YOLO("/root/autodl-tmp/ultralytics/runs/pose/train55/weights/best.pt")  # 使用第六次訓練的最佳權重

# 專注於提高pose mAP50-95的訓練策略
results = model.train(
    data="coco-pose.yaml",
    epochs=50,            # 減少epochs數量以節省訓練時間
    imgsz=1536,           # 進一步提高解析度以提升精度
    batch=32,             # 降低批次大小以適應更高解析度
    save_period=1,        # 每個epoch保存
    cache="disk",         # 使用磁盤緩存
    optimizer="AdamW",    # 繼續使用AdamW優化器
    lr0=0.000008,         # 更低的學習率進行精細微調
    lrf=0.001,            # 更小的最終學習率因子
    cos_lr=True,          # 餘弦學習率調度
    warmup_epochs=3.0,    # 較長的熱身期
    device="0,1,2,3",     # 使用全部4個GPU
    patience=40,          # 增加耐心值
    box=10.0,             # 調整框損失權重
    pose=30.0,            # 進一步提高姿態損失權重
    kobj=8.0,             # 增加關鍵點對象損失權重
    close_mosaic=0,       # 完全關閉馬賽克增強
    amp=True,             # 啟用混合精度訓練
    overlap_mask=True,    # 啟用重疊口罩
    
    # 數據增強參數 (根據官方文檔設置)
    hsv_h=0.01,           # 色調變化 (0.0-1.0)
    hsv_s=0.1,            # 飽和度變化 (0.0-1.0)
    hsv_v=0.1,            # 亮度變化 (0.0-1.0)
    degrees=0.0,          # 旋轉增強 (0.0-180.0)
    translate=0.05,       # 平移增強 (0.0-1.0)
    scale=0.05,           # 縮放增強 (>=0.0)
    shear=0.0,            # 剪切增強 (-180.0-180.0)
    perspective=0.0,      # 透視變換 (0.0-0.001)
    flipud=0.0,           # 垂直翻轉概率 (0.0-1.0)
    fliplr=0.5,           # 水平翻轉概率 (0.0-1.0)
    mosaic=0.0,           # 馬賽克增強概率 (0.0-1.0)
    mixup=0.0,            # Mixup增強概率 (0.0-1.0)
    copy_paste=0.0        # 複製粘貼增強概率 (0.0-1.0)
) 