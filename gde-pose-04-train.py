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
    # 載入第三階段訓練的最佳模型
    model = YOLO("gde_pose_distill/gde_distill_phase3/weights/best.pt")  # 使用第三階段最佳權重

    # 訓練模型 - 第四階段：最終性能微調和實際應用場景適應
    results = model.train(
        # 基本訓練設置
        data="coco-pose.yaml",
        epochs=30,               # 僅訓練30個epoch進行最終調整
        imgsz=640,
        batch=96,                # 進一步降低批次大小，提高精度
        device=[0, 1, 2, 3],     # 使用4張4090
        workers=16,              # 最大化資料加載
        
        # 蒸餾參數 - 最終階段專注於模型本身優化
        teacher="yolo11n-pose.pt",
        # 僅保留高層進行最終蒸餾，關注語義特徵
        target_layers=["model.7.conv"],
        distill=0.5,             # 大幅降低蒸餾權重
        freezeAllBN=True,        # 保持BN層凍結
        freeze=[0, 1, 3],        # 凍結前三層卷積層，保留已學習的特徵
        
        # 損失函數權重 - 最終性能微調階段
        pose=10.0,               # 平衡姿態損失權重
        kobj=1.2,                # 降低關鍵點目標性損失權重
        box=4.0,                 # 降低框損失權重
        cls=0.3,                 # 降低分類損失權重
        
        # 優化器設置 - 精確微調
        optimizer="AdamW",       
        lr0=0.0001,              # 極低學習率，僅進行最小調整
        lrf=0.0001,              # 幾乎平坦的學習率衰減
        momentum=0.98,           # 最高動量，確保方向穩定
        weight_decay=0.0002,     # 精細調整權重衰減
        
        # 訓練策略 - 最終微調階段
        warmup_epochs=0.0,       # 無預熱期
        cos_lr=True,             # 保持余弦學習率調度
        close_mosaic=0,          # 完全關閉mosaic增強
        patience=10,             # 進一步降低早停耐心
        
        # 數據增強 - 最終階段幾乎無增強，專注真實數據
        hsv_h=0.0,               # 無色調增強
        hsv_s=0.1,               # 最小飽和度增強
        hsv_v=0.1,               # 最小亮度增強
        degrees=0.0,             # 無旋轉增強
        translate=0.02,          # 最小平移
        scale=0.1,               # 最小縮放
        shear=0.0,               # 無剪切
        fliplr=0.1,              # 最小翻轉概率
        mosaic=0.0,              # 關閉Mosaic
        mixup=0.0,               # 無mixup
        
        # 模型優化
        # optimize=True,           # 增加模型優化
        
        # 保存與評估
        save_period=1,           # 每個epoch都保存
        val=True,
        name="gde_distill_phase4_final",
        project="gde_pose_distill",
        exist_ok=True,
        
        # 性能相關
        amp=True,                # 使用混合精度加速
        
        # 穩定性設置
        seed=42,                 # 保持隨機種子一致
        deterministic=True,
        
        # 最終模型導出設置
        # export=True,             # 自動導出最終模型
    )

if __name__ == "__main__":
    main() 