from ultralytics import YOLO

# 加載第七次訓練的最佳權重
model = YOLO("/root/autodl-tmp/ultralytics/runs/pose/train55/weights/best.pt")  # 使用最佳權重

# 最終微調策略：根據官方參數優化
results = model.train(
    data="coco-pose.yaml",
    epochs=100,             # 延長訓練週期
    imgsz=1280,             # 保持較高解析度
    batch=32,               # 使用中等批次大小
    save_period=1,          # 每個epoch保存
    cache="disk",           # 使用磁盤緩存
    optimizer="AdamW",      # 繼續使用AdamW優化器
    lr0=0.00001,            # 非常低的學習率
    lrf=0.01,               # 標準最終學習率因子
    momentum=0.85,          # 降低動量值以精確優化
    weight_decay=0.0001,    # 減少權重衰減以減少正則化
    warmup_epochs=0.0,      # 關閉熱身
    warmup_momentum=0.8,    # 設置熱身動量
    warmup_bias_lr=0.01,    # 設置熱身偏置學習率
    box=6.0,                # 降低框損失權重
    pose=25.0,              # 提高姿態損失權重
    kobj=7.0,               # 提高關鍵點對象損失權重
    cls=0.2,                # 降低分類損失權重 
    dfl=1.0,                # 降低分布焦點損失權重
    nbs=64,                 # 標準標稱批次大小
    cos_lr=True,            # 啟用餘弦學習率調度
    close_mosaic=0,         # 完全關閉馬賽克增強
    amp=True,               # 啟用混合精度訓練
    device="0,1,2,3",       # 使用全部4個GPU
    dropout=0.2,            # 添加dropout正則化
    overlap_mask=True,      # 啟用重疊口罩
    patience=50,            # 較長耐心值
    val=True,               # 確保每個epoch驗證
    freeze=15,              # 凍結前15層，專注於微調後層
    
    # 數據增強參數調整為更適合關鍵點精度的值
    hsv_h=0.01,             # 最小色調變化
    hsv_s=0.1,              # 減少飽和度變化
    hsv_v=0.1,              # 減少亮度變化
    degrees=0.0,            # 關閉旋轉
    translate=0.05,         # 最小平移
    scale=0.1,              # 最小縮放
    fliplr=0.5,             # 保持水平翻轉
    mosaic=0.0,             # 關閉馬賽克
    mixup=0.0,              # 關閉mixup
    copy_paste=0.0          # 關閉複製粘貼
)
