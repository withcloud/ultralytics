import os
import sys
import torch
import torch.distributed as dist
import warnings
from ultralytics import YOLO
from ultralytics.utils import LOGGER
from ultralytics.utils.callbacks.base_callbacks import (
    on_pretrain_routine_start,
    on_train_start,
    on_train_epoch_start,
    on_train_batch_start,
    on_train_batch_end,
    on_train_epoch_end,
    on_val_start,
    on_val_batch_start,
    on_val_batch_end,
    on_val_end,
)

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

# 定義回調函數來記錄每個 GPU 的信息

@on_pretrain_routine_start
def log_pretraining_start(trainer):
    """在預訓練例程開始時從所有 GPU 記錄信息"""
    rank = dist.get_rank() if dist.is_initialized() else 0
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    gpu_name = torch.cuda.get_device_name(local_rank) if torch.cuda.is_available() else "CPU"
    print(f"[Rank {rank}, GPU {local_rank}] Starting pre-training routine on {gpu_name}")
    if dist.is_initialized():
        dist.barrier()

@on_train_start
def log_train_start(trainer):
    """在訓練開始時從所有 GPU 記錄信息"""
    rank = dist.get_rank() if dist.is_initialized() else 0
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    print(f"[Rank {rank}, GPU {local_rank}] Starting training")
    if dist.is_initialized():
        dist.barrier()

@on_train_epoch_start
def log_epoch_start(trainer):
    """在每個 epoch 開始時從所有 GPU 記錄信息"""
    rank = dist.get_rank() if dist.is_initialized() else 0
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    print(f"[Rank {rank}, GPU {local_rank}] Starting epoch {trainer.epoch}")
    if dist.is_initialized():
        dist.barrier()

@on_train_batch_start
def log_batch_start(trainer):
    """在每個批次開始時從所有 GPU 記錄信息"""
    # 批次開始時日誌會很多，所以我們只在每 10 個批次記錄一次
    if trainer.batch_idx % 10 == 0:
        rank = dist.get_rank() if dist.is_initialized() else 0
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        print(f"[Rank {rank}, GPU {local_rank}] Batch {trainer.batch_idx} starting")
        if dist.is_initialized():
            dist.barrier()

@on_train_batch_end
def log_batch_end(trainer):
    """在每個批次結束時從所有 GPU 記錄信息"""
    # 批次結束時日誌會很多，所以我們只在每 10 個批次記錄一次
    if trainer.batch_idx % 10 == 0:
        rank = dist.get_rank() if dist.is_initialized() else 0
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        loss = trainer.loss.item() if hasattr(trainer.loss, 'item') else trainer.loss
        print(f"[Rank {rank}, GPU {local_rank}] Batch {trainer.batch_idx} completed, Loss: {loss:.4f}")
        if dist.is_initialized():
            dist.barrier()

@on_train_epoch_end
def log_epoch_end(trainer):
    """在每個 epoch 結束時從所有 GPU 記錄信息"""
    rank = dist.get_rank() if dist.is_initialized() else 0
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    loss = trainer.loss.item() if hasattr(trainer.loss, 'item') else trainer.loss
    print(f"[Rank {rank}, GPU {local_rank}] Epoch {trainer.epoch} completed, Loss: {loss:.4f}")
    if dist.is_initialized():
        dist.barrier()

@on_val_start
def log_val_start(validator):
    """在驗證開始時從所有 GPU 記錄信息"""
    rank = dist.get_rank() if dist.is_initialized() else 0
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    print(f"[Rank {rank}, GPU {local_rank}] Starting validation")
    if dist.is_initialized():
        dist.barrier()

@on_val_batch_start
def log_val_batch_start(validator):
    """在每個驗證批次開始時從所有 GPU 記錄信息"""
    # 驗證批次開始時日誌會很多，所以我們只在每 5 個批次記錄一次
    if validator.batch_idx % 5 == 0:
        rank = dist.get_rank() if dist.is_initialized() else 0
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        print(f"[Rank {rank}, GPU {local_rank}] Validation batch {validator.batch_idx} starting")
        if dist.is_initialized():
            dist.barrier()

@on_val_batch_end
def log_val_batch_end(validator):
    """在每個驗證批次結束時從所有 GPU 記錄信息"""
    # 驗證批次結束時日誌會很多，所以我們只在每 5 個批次記錄一次
    if validator.batch_idx % 5 == 0:
        rank = dist.get_rank() if dist.is_initialized() else 0
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        print(f"[Rank {rank}, GPU {local_rank}] Validation batch {validator.batch_idx} completed")
        if dist.is_initialized():
            dist.barrier()

@on_val_end
def log_val_end(validator):
    """在驗證結束時從所有 GPU 記錄信息"""
    rank = dist.get_rank() if dist.is_initialized() else 0
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    print(f"[Rank {rank}, GPU {local_rank}] Validation completed")
    if dist.is_initialized():
        dist.barrier()

def print_from_all_processes(message):
    """確保所有進程打印信息"""
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    print(f"[Rank {local_rank}, GPU {local_rank}] {message}")
    
    if world_size > 1 and dist.is_initialized():
        dist.barrier()  # 同步所有進程

def main():
    # 打印訓練開始信息
    gpu_name = torch.cuda.get_device_name() if torch.cuda.is_available() else "CPU"
    print_from_all_processes(f"Starting training on GPU {gpu_name}")

    # 載入模型
    model = YOLO("yolo11n-pose.pt")
    
    # 註冊所有回調函數
    model.add_callback("on_pretrain_routine_start", log_pretraining_start)
    model.add_callback("on_train_start", log_train_start)
    model.add_callback("on_train_epoch_start", log_epoch_start)
    model.add_callback("on_train_batch_start", log_batch_start)
    model.add_callback("on_train_batch_end", log_batch_end)
    model.add_callback("on_train_epoch_end", log_epoch_end)
    model.add_callback("on_val_start", log_val_start)
    model.add_callback("on_val_batch_start", log_val_batch_start)
    model.add_callback("on_val_batch_end", log_val_batch_end)
    model.add_callback("on_val_end", log_val_end)

    # 訓練模型
    results = model.train(
        data="coco8-pose.yaml",
        epochs=20,
        imgsz=640,
        device=[0, 1],
        teacher="yolo11n-pose.pt",
        target_layers=["model.0.conv", 1],
        verbose=True,
        # freezeAllBN=True,
        # freeze=23,
    )

if __name__ == "__main__":
    main()