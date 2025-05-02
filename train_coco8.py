import os
import torch
import torch.distributed as dist
from ultralytics import YOLO
from ultralytics.utils import LOGGER
import sys
import warnings

# 強制所有進程輸出日誌
os.environ["RANK"] = "-1"

# 設置環境變量，強制所有進程輸出日誌
os.environ["RANK"] = "-1"  # 覆蓋 rank 檢查
LOGGER.setLevel('INFO')  # 設置日誌級別

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


def on_pretrain_routine_start(trainer):
    """在預訓練開始時從所有 GPU 記錄信息"""
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    gpu_name = torch.cuda.get_device_name(local_rank) if torch.cuda.is_available() else "CPU"
    print(f"[Rank {local_rank}, GPU {local_rank}] Starting pre-training on {gpu_name}")


def on_train_epoch_start(trainer):
    """在每個訓練 epoch 開始時從所有 GPU 記錄信息"""
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    print(f"[Rank {local_rank}, GPU {local_rank}] Starting epoch {trainer.epoch}")


def on_train_batch_end(trainer):
    """在每個訓練批次結束時從所有 GPU 記錄信息"""
    # 每 10 個批次記錄一次，避免日誌過多
    if trainer.batch_idx % 10 == 0:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        loss = trainer.loss.item() if hasattr(trainer.loss, 'item') else trainer.loss
        print(f"[Rank {local_rank}, GPU {local_rank}] Batch {trainer.batch_idx}, Loss: {loss:.4f}")


def on_val_start(validator):
    """在驗證開始時從所有 GPU 記錄信息"""
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    print(f"[Rank {local_rank}, GPU {local_rank}] Starting validation")


def main():
    # 載入模型
    model = YOLO("yolo11n-pose.pt")
    
    # 按照官方用法添加回調
    model.add_callback("on_pretrain_routine_start", on_pretrain_routine_start)
    model.add_callback("on_train_epoch_start", on_train_epoch_start)
    model.add_callback("on_train_batch_end", on_train_batch_end)
    model.add_callback("on_val_start", on_val_start)

    # 訓練模型
    results = model.train(
        data="coco8-pose.yaml",
        epochs=20,
        imgsz=640,
        # device=[0, 1],
        teacher="yolo11n-pose.pt",
        target_layers=["model.0.conv", 1],
        verbose=True,
    )


if __name__ == "__main__":
    main()