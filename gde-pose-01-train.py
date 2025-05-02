import os
import torch
import torch.nn.functional as F
from ultralytics import YOLO
from ultralytics.utils import LOGGER

import sys
import torch.nn as nn
import torch.distributed as dist
import warnings

# 添加本地路徑到 Python 路徑中，確保使用本地版本
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, current_dir)
sys.path.insert(0, parent_dir)

# 設置環境變量，確保分佈式訓練使用本地代碼
os.environ["PYTHONPATH"] = f"{current_dir}:{os.environ.get('PYTHONPATH', '')}"
# 忽略 DDP 的 stride 不匹配警告
warnings.filterwarnings("ignore", message="Grad strides do not match bucket view strides")
# 忽略除零警告
warnings.filterwarnings("ignore", message="divide by zero encountered in divide")

def main():
    # Load a model
    model = YOLO("models/gde-pose-640.pt")

    # Train the model
    results = model.train(
        # 基本訓練設置
        data="coco-pose.yaml",
        epochs=100,             # 初步實驗用20個epoch
        imgsz=640,
        batch=126,             # 4個4090的大批量
        device=[0, 1, 2, 3, 4, 5],
        workers=16,
        
        # 蒸餾參數
        teacher="yolo11n-pose.pt",
        target_layers=["model.0.conv", "model.1.conv"],
        distill=0.8,           # 較高蒸餾權重
        freezeAllBN=True,

        # box=0.00001, # (float) box loss gain
        # cls=0.00001, # (float) cls loss gain (scale with pixels)
        # dfl=0.00001, # (float) dfl loss gain
        # pose=0.00001, # (float) pose loss gain
        # kobj=0.00001, # (float) keypoint obj loss gain
        
        # 優化器設置
        optimizer="AdamW",    # AdamW通常更穩定
        lr0=0.001,            # 初始學習率
        lrf=0.01,             # 最終學習率因子
        momentum=0.937,       # 動量參數
        weight_decay=0.0005,  # 權重衰減
        
        # 訓練策略
        warmup_epochs=3.0,    # 預熱epochs
        cos_lr=True,          # 使用余弦學習率調度
        close_mosaic=10,      # 最後10個epoch關閉mosaic
        
        # 數據增強設置
        hsv_h=0.015,          # 色調變化
        hsv_s=0.7,            # 飽和度變化
        hsv_v=0.4,            # 亮度變化
        degrees=10.0,         # 旋轉角度範圍
        translate=0.1,        # 平移比例
        scale=0.5,            # 縮放比例
        shear=2.0,            # 剪切角度
        fliplr=0.5,           # 左右翻轉
        mosaic=1.0,           # 使用mosaic增強
        
        # 保存與評估
        save_period=10,       # 每10個epoch保存一次
        val=10,               # 每10個epoch驗證一次
        name="gde_distill_01_layers",
        project="gde_pose_distill",
        exist_ok=True,
        
       # 性能相關
        amp=True,             # 混合精度訓練
        
        # 早停策略
        patience=20,          # 20個epoch無改善則早停
        
        # 穩定性設置
        seed=42,              # 固定隨機種子
        deterministic=True,   # 確保結果可複現
    )

if __name__ == "__main__":
    main()
