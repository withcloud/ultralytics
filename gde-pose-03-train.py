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
    # 載入第二階段訓練的最佳模型
    model = YOLO("gde_pose_distill/gde_distill_phase2/weights/best.pt")  # 使用第二階段最佳權重

    # 訓練模型 - 第三階段專注於微調和最終性能
    results = model.train(
        # 基本訓練設置
        data="coco-pose.yaml",
        epochs=50,               # 再訓練50個epoch進行微調
        imgsz=640,
        batch=128,               # 適度降低批次大小，增加穩定性
        device=[0, 1, 2, 3],     # 使用4張4090
        workers=16,              # 最大化資料加載
        
        # 蒸餾參數 - 最終微調階段，專注於細緻層面
        teacher="yolo11n-pose.pt",
        # 第三階段專注於中間層和高層的特徵精煉
        target_layers=["model.3.conv", "model.5.conv", "model.7.conv"],
        distill=0.7,             # 降低蒸餾權重，給予模型更多自主學習空間
        freezeAllBN=True,        # 保持BN層凍結
        freeze=[0, 1],           # 凍結前兩層，保持之前學到的低層特徵
        
        # 損失函數權重 - 最終微調階段
        pose=12.0,               # 適度降低姿態損失權重
        kobj=1.5,                # 進一步調整關鍵點目標性損失權重
        box=5.0,                 # 輕微降低框損失權重
        
        # 優化器設置 - 最終微調階段
        optimizer="AdamW",       
        lr0=0.0003,              # 更低的學習率進行精細調整
        lrf=0.001,               # 更平緩的學習率衰減
        momentum=0.97,           # 增加動量，保持方向穩定性
        weight_decay=0.0003,     # 增加權重衰減防止過擬合
        
        # 訓練策略 - 最終微調階段
        warmup_epochs=1.0,       # 極短預熱期
        cos_lr=True,             # 保持余弦學習率調度
        close_mosaic=0,          # 第三階段完全關閉mosaic增強
        patience=15,             # 降低早停耐心，避免無意義的訓練
        
        # 數據增強 - 最終微調階段減少增強強度
        hsv_h=0.005,             # 最小顏色增強
        hsv_s=0.3,               
        hsv_v=0.2,               
        degrees=5.0,             # 減少旋轉增強
        translate=0.05,          # 減少平移
        scale=0.3,               # 減少縮放範圍
        shear=1.0,               # 減少剪切
        fliplr=0.3,              # 減少翻轉機率
        mosaic=0.0,              # 關閉Mosaic
        mixup=0.0,               # 保持無mixup
        
        # 保存與評估
        save_period=1,           # 每個epoch都保存
        val=True,
        name="gde_distill_phase3",
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