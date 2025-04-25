from ultralytics import YOLO

# 加載第七次訓練的最佳權重
model = YOLO("/root/autodl-tmp/ultralytics/runs/pose/train/weights/best.pt")  # 使用第七次訓練的最佳權重

# 針對COCO評估標準的優化策略
results = model.train(
    data="coco-pose.yaml",
    epochs=30,            # 增加訓練時間以確保充分收斂
    imgsz=1280,           # 使用標準COCO評估解析度
    batch=32,             # 使用較小批次以提高精度
    save_period=1,        # 每個epoch保存
    cache="disk",         # 使用磁盤緩存
    optimizer="AdamW",    # 繼續使用AdamW優化器
    lr0=0.000003,         # 極低學習率進行精細微調
    lrf=0.0001,           # 非常小的最終學習率因子
    cos_lr=True,          # 餘弦學習率調度
    warmup_epochs=0.0,    # 關閉熱身
    device="0,1,2,3",     # 使用全部4個GPU
    patience=20,          # 較長的耐心值
    box=7.0,              # 降低框損失權重
    pose=50.0,            # 極高的姿態損失權重
    kobj=15.0,            # 大幅增加關鍵點對象損失權重
    nbs=32,               # 標稱批次大小
    close_mosaic=0,       # 完全關閉馬賽克增強
    amp=True,             # 啟用混合精度訓練
    overlap_mask=True,    # 啟用重疊口罩
    val=True,             # 確保每個epoch進行驗證
    fliplr=0.5            # 水平翻轉概率
)
