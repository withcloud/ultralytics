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
        epochs=180,             
        imgsz=640,
        batch=160,              # 適度增加批量
        device=[0, 1, 2, 3],
        workers=16,
        
        # 蒸餾參數 - 更激進版本
        teacher="yolo11n-pose.pt",
        target_layers=["model.0.conv", "model.1.conv", "model.2.cv1.conv", "model.2.cv2.conv", "model.3.conv", "model.4.cv1.conv", "model.4.cv2.conv", "model.5.conv", "model.6.cv1.conv", "model.6.cv2.conv", "model.7.conv"],
        distill=0.9,            # 增加蒸餾權重
        freezeAllBN=False,

        # 損失函數權重 - 更激進版本
        pose=18.0,              # 增加姿態損失權重
        kobj=2.5,               # 增加關鍵點目標性損失權重
        
        # 優化器設置 - 更激進版本
        optimizer="AdamW",      
        lr0=0.0015,             # 更高的學習率
        lrf=0.01,              
        momentum=0.937,        
        weight_decay=0.0005,   
        
        # 訓練策略 - 更激進版本
        warmup_epochs=3.0,      # 縮短預熱時間
        cos_lr=True,            
        close_mosaic=15,        # 延後關閉mosaic
        
        # 數據增強 - 更激進版本
        hsv_h=0.02,             # 略微增強顏色增強
        hsv_s=0.8,             
        hsv_v=0.5,             
        degrees=12.0,           # 增強旋轉增強
        translate=0.12,        
        scale=0.6,              # 更大的縮放範圍
        shear=2.5,             
        fliplr=0.5,            
        mosaic=1.0,            
        mixup=0.05,             # 添加輕微mixup
        
        # 保存與評估
        save_period=2,          # 降低保存頻率以提高速度
        val=True,
        name="gde_distill_aggressive",
        project="gde_pose_distill",
        exist_ok=True,
        
        # 性能相關
        amp=True,              
        
        # 早停策略
        patience=25,            # 調整早停耐心
        
        # 穩定性設置
        seed=42,               
        deterministic=True,    
    )

if __name__ == "__main__":
    main()
