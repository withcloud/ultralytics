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
        epochs=20,             # 初步實驗用20個epoch
        imgsz=640,
        batch=126,             # 4個4090的大批量
        device=[0, 1, 2, 3, 4, 5],
        workers=16,
        
        # 蒸餾參數
        teacher="yolo11n-pose.pt",
        target_layers=["model.0.conv", "model.1.conv"],
        distill=0.0001,           # 較高蒸餾權重
        freezeAllBN=True,

        box=0.0, # (float) box loss gain
        cls=0.0, # (float) cls loss gain (scale with pixels)
        dfl=0.0, # (float) dfl loss gain
        pose=0.0, # (float) pose loss gain
        kobj=0.0, # (float) keypoint obj loss gain
        
        # 優化器設置
        optimizer="Adam",
        lr0=0.001,             # 適中的學習率
        lrf=0.1,               # 學習率可以衰減更多
        weight_decay=0.0005,
        warmup_epochs=2,
        
        # 保存與評估
        save_period=1,
        val=True,
        amp=True,
        
        # 實驗命名
        name="gde_distill_01_layers",
        project="gde_pose_distill",
        exist_ok=True
    )

if __name__ == "__main__":
    main()
