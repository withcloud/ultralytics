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
    # Load a model - using the checkpoint from the first distillation phase
    model = YOLO("gde_pose_distill/gde_distill_aggressive/weights/best.pt")  # 使用第一階段最佳權重

    # Train the model
    results = model.train(
        # 基本訓練設置
        data="coco-pose.yaml",
        epochs=100,              # 減少總訓練週期
        imgsz=640,
        batch=160,               # 保持批量大小
        device=[0, 1, 2, 3],     # 使用4張4090
        workers=16,              # 最大化資料加載
        
        # 蒸餾參數 - 優化版本
        teacher="yolo11n-pose.pt",
        target_layers=["model.0.conv", "model.1.conv", "model.3.conv", "model.5.conv", "model.7.conv"],
        distill=1.0,             # 增加蒸餾權重，較第一階段更高
        freezeAllBN=True,        # 凍結BN層以穩定訓練
        
        # 損失函數權重 - 優化版本
        pose=15.0,               # 輕微降低姿態損失權重
        kobj=1.8,                # 調整關鍵點目標性損失權重
        box=6.0,                 # 降低框損失權重
        
        # 優化器設置 - 優化版本
        optimizer="AdamW",       
        lr0=0.0008,              # 降低學習率，更適合微調
        lrf=0.005,               # 更陡的學習率下降
        momentum=0.95,           # 增加動量
        weight_decay=0.0004,     # 微調權重衰減
        
        # 訓練策略 - 優化版本
        warmup_epochs=2.0,       # 縮短預熱時間
        cos_lr=True,             
        close_mosaic=10,         # 提前關閉mosaic
        patience=20,             # 提早停止的容忍度降低
        
        # 數據增強 - 減少擾動
        hsv_h=0.01,              # 減少顏色增強
        hsv_s=0.5,               
        hsv_v=0.3,               
        degrees=10.0,            # 減少旋轉增強
        translate=0.1,           
        scale=0.4,               # 減少縮放範圍
        shear=2.0,               
        fliplr=0.5,              
        mosaic=0.8,              # 降低Mosaic概率
        mixup=0.0,               # 移除mixup
        
        # 保存與評估
        save_period=1,           # 更頻繁保存以追蹤進度
        val=True,
        name="gde_distill_phase2",
        project="gde_pose_distill",
        exist_ok=True,
        
        # 性能相關
        amp=True,                # 使用混合精度加速
        
        # 穩定性設置
        seed=42,                 # 保持隨機種子一致
        deterministic=True,      
    )

if __name__ == "__main__":
    main() 