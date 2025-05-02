import os
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
import time

# 簡單的資料集
class DummyDataset(Dataset):
    def __init__(self, size=1000):
        self.size = size
        self.data = torch.randn(size, 20)
        self.targets = torch.randint(0, 2, (size,))
        
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

# 簡單的模型
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(20, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )
        
    def forward(self, x):
        return self.layers(x)

# 為每個進程設置分佈式環境
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # 初始化進程組
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

# 清理分佈式環境
def cleanup():
    dist.destroy_process_group()

# 每個 GPU 執行的訓練函數
def train(rank, world_size):
    # 初始化分佈式環境
    setup(rank, world_size)
    
    # 打印當前進程信息，確保所有 GPU 都有日誌
    print(f"[GPU {rank}] 進程初始化完成，世界大小: {world_size}")
    
    # 創建模型並移至對應 GPU
    model = SimpleModel().to(rank)
    # 轉換為 DDP 模型
    ddp_model = DDP(model, device_ids=[rank])
    
    # 打印模型位置信息
    print(f"[GPU {rank}] 模型已創建並放置於 device {rank}")
    
    # 創建資料集和資料載入器
    dataset = DummyDataset(1000)
    # 使用 DistributedSampler 來分配資料給不同進程
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, batch_size=32, sampler=sampler)
    
    print(f"[GPU {rank}] 資料載入器已創建，批次大小: 32，批次數: {len(dataloader)}")
    
    # 定義損失函數和優化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(ddp_model.parameters(), lr=0.01)
    
    # 訓練迴圈
    epochs = 3
    for epoch in range(epochs):
        # 設置 sampler 的 epoch 屬性，確保洗牌不同
        sampler.set_epoch(epoch)
        
        # 開始時間
        start_time = time.time()
        
        # 記錄總損失
        total_loss = 0.0
        
        # 同步進程進入訓練階段
        print(f"[GPU {rank}] Epoch {epoch+1}/{epochs} 開始訓練")
        if dist.is_initialized():
            dist.barrier()
        
        # 一個 epoch 的訓練迴圈
        for i, (inputs, targets) in enumerate(dataloader):
            # 將資料移至對應 GPU
            inputs, targets = inputs.to(rank), targets.to(rank)
            
            # 前向傳播
            outputs = ddp_model(inputs)
            loss = criterion(outputs, targets)
            
            # 反向傳播和優化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 累計損失
            total_loss += loss.item()
            
            # 每 10 個批次打印一次
            if (i + 1) % 10 == 0:
                print(f"[GPU {rank}] Epoch {epoch+1}, Batch {i+1}/{len(dataloader)}, Loss: {loss.item():.4f}")
        
        # 計算平均損失
        avg_loss = total_loss / len(dataloader)
        # 計算訓練時間
        epoch_time = time.time() - start_time
        
        # 每個 epoch 結束時打印統計信息
        print(f"[GPU {rank}] Epoch {epoch+1} 完成, 平均損失: {avg_loss:.4f}, 訓練時間: {epoch_time:.2f} 秒")
        
        # 同步所有進程在 epoch 結束
        if dist.is_initialized():
            dist.barrier()
    
    # 清理
    cleanup()

def main():
    # 獲取可用的 GPU 數量
    world_size = torch.cuda.device_count()
    print(f"發現 {world_size} 個 GPU")
    
    if world_size < 2:
        print("需要至少 2 個 GPU 來演示多 GPU 訓練")
        return
    
    # 使用 mp.spawn 同時啟動多個進程
    mp.spawn(
        train,
        args=(world_size,),
        nprocs=world_size,
        join=True
    )

if __name__ == "__main__":
    main()