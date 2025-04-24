from ultralytics import YOLO

# Load the model from the fifth training stage best weights
model = YOLO("/root/autodl-tmp/ultralytics/runs/pose/train55/weights/best.pt")  # 使用第五次訓練的最佳權重

# Train the model with focus on pose keypoint precision
results = model.train(
    data="coco-pose.yaml",
    epochs=50,                 # 減少訓練週期，更適合精調階段
    imgsz=1600,                # 增加解析度以更好地捕捉細節
    batch=64,                  # 增加批次大小以充分利用GPU內存
    save_period=1,             # 每個epoch保存一次
    cache="disk",              # 使用磁盤緩存
    optimizer="AdamW",         # 繼續使用AdamW優化器
    lr0=0.00001,               # 更低的學習率進行精細調整
    lrf=0.002,                 # 更低的最終學習率比例
    cos_lr=True,               # 餘弦學習率調整
    warmup_epochs=2.0,         # 縮短熱身時間，與總週期相適應
    device="0,1,2,3",          # 使用所有GPU
    patience=25,               # 調整耐心值，與總週期相適應
    box=15.0,                  # 增加框損失權重
    pose=30.0,                 # 大幅增加姿態損失權重
    kobj=8.0,                  # 增加關鍵點對象損失權重
    close_mosaic=0,            # 完全關閉馬賽克增強
    amp=True,                  # 自動混合精度
    nbs=64,                    # 標稱批次大小調整為與batch一致
    overlap_mask=True,         # 掩碼重疊
    weight_decay=0.001,        # 增加權重衰減以提高泛化能力
    dropout=0.03,              # 增加dropout以防過擬合
    val=True,                  # 進行驗證
    plots=True,                # 生成訓練圖表
    degrees=0.0,               # 關閉旋轉增強
    translate=0.1,             # 保留平移增強
    scale=0.1,                 # 保留縮放增強
    hsv_h=0.015,               # 輕微色調增強
    hsv_s=0.2,                 # 適當的飽和度增強
    hsv_v=0.2                  # 適當的亮度增強
) 



