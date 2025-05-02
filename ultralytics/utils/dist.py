# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import os
import shutil
import socket
import sys
import tempfile
import logging
from pathlib import Path
import torch.distributed as dist

from . import USER_CONFIG_DIR
from .torch_utils import TORCH_1_9


def find_free_network_port() -> int:
    """
    Find a free port on localhost.

    It is useful in single-node training when we don't want to connect to a real main node but have to set the
    `MASTER_PORT` environment variable.

    Returns:
        (int): The available network port number.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]  # port


def generate_ddp_file(trainer):
    """
    Generate a DDP (Distributed Data Parallel) file for multi-GPU training.

    This function creates a temporary Python file that enables distributed training across multiple GPUs.
    The file contains the necessary configuration to initialize the trainer in a distributed environment.

    Args:
        trainer (object): The trainer object containing training configuration and arguments.
                         Must have args attribute and be a class instance.

    Returns:
        (str): Path to the generated temporary DDP file.

    Notes:
        The generated file is saved in the USER_CONFIG_DIR/DDP directory and includes:
        - Trainer class import
        - Configuration overrides from the trainer arguments
        - Model path configuration
        - Training initialization code
    """
    module, name = f"{trainer.__class__.__module__}.{trainer.__class__.__name__}".rsplit(".", 1)

    content = f"""
# Ultralytics Multi-GPU training temp file (should be automatically deleted after use)
import os
import sys
import logging
from pathlib import Path
import torch.distributed as dist

# 設置所有 GPU 進程顯示日誌，無視 rank
overrides = {vars(trainer.args)}

if __name__ == "__main__":
    from {module} import {name}
    from ultralytics.utils import DEFAULT_CFG_DICT, LOGGER
    import builtins
    
    # 強制所有 GPU 進程顯示日誌
    rank = int(os.environ.get("RANK", -1))
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    
    # 打印函數重定向，確保所有進程輸出
    original_print = builtins.print
    def rank_print(*args, **kwargs):
        gpu_id = local_rank if local_rank != -1 else 0
        prefix = f"[Rank {{rank}}, GPU {{gpu_id}}] "
        original_print(prefix, *args, **kwargs)
    builtins.print = rank_print
    
    # 設置所有進程顯示 INFO 級別日誌
    root_logger = logging.getLogger()
    # 移除所有現有的處理器
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # 添加新的處理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    root_logger.setLevel(logging.INFO)
    
    # 重新配置 LOGGER
    for handler in LOGGER.handlers[:]:
        LOGGER.removeHandler(handler)
    LOGGER.addHandler(console_handler)
    LOGGER.setLevel(logging.INFO)
    
    # 添加 rank 信息到各個進程的日誌前綴
    gpu_id = local_rank if local_rank != -1 else 0
    log_prefix = f"[Rank {{rank}}, GPU {{gpu_id}}] "
    
    # 輸出啟動信息
    print(f"{{log_prefix}}DDP 進程啟動，RANK={{rank}}, LOCAL_RANK={{local_rank}}")
    
    # 確保所有進程都初始化完畢
    if rank != -1:
        try:
            dist.barrier()
            print(f"{{log_prefix}}所有進程同步完畢，開始訓練")
        except Exception as e:
            print(f"{{log_prefix}}進程同步失敗: {{e}}")
    
    cfg = DEFAULT_CFG_DICT.copy()
    cfg.update(save_dir='')   # handle the extra key 'save_dir'
    trainer = {name}(cfg=cfg, overrides=overrides)
    trainer.args.model = "{getattr(trainer.hub_session, "model_url", trainer.args.model)}"
    
    # 修改 trainer 中的日誌設置，確保所有進程輸出日誌
    original_info = LOGGER.info
    def rank_info(msg, *args, **kwargs):
        gpu_id = local_rank if local_rank != -1 else 0
        prefix = f"[Rank {{rank}}, GPU {{gpu_id}}] "
        original_info(f"{{prefix}}{{msg}}", *args, **kwargs)
    LOGGER.info = rank_info
    
    results = trainer.train()
"""
    (USER_CONFIG_DIR / "DDP").mkdir(exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix="_temp_",
        suffix=f"{id(trainer)}.py",
        mode="w+",
        encoding="utf-8",
        dir=USER_CONFIG_DIR / "DDP",
        delete=False,
    ) as file:
        file.write(content)
    return file.name


def generate_ddp_command(world_size, trainer):
    """
    Generate command for distributed training.

    Args:
        world_size (int): Number of processes to spawn for distributed training.
        trainer (object): The trainer object containing configuration for distributed training.

    Returns:
        cmd (List[str]): The command to execute for distributed training.
        file (str): Path to the temporary file created for DDP training.
    """
    import __main__  # noqa local import to avoid https://github.com/Lightning-AI/pytorch-lightning/issues/15218

    if not trainer.resume:
        shutil.rmtree(trainer.save_dir)  # remove the save_dir
    file = generate_ddp_file(trainer)
    dist_cmd = "torch.distributed.run" if TORCH_1_9 else "torch.distributed.launch"
    port = find_free_network_port()
    cmd = [sys.executable, "-m", dist_cmd, "--nproc_per_node", f"{world_size}", "--master_port", f"{port}", file]
    return cmd, file


def ddp_cleanup(trainer, file):
    """
    Delete temporary file if created during distributed data parallel (DDP) training.

    This function checks if the provided file contains the trainer's ID in its name, indicating it was created
    as a temporary file for DDP training, and deletes it if so.

    Args:
        trainer (object): The trainer object used for distributed training.
        file (str): Path to the file that might need to be deleted.

    Examples:
        >>> trainer = YOLOTrainer()
        >>> file = "/tmp/ddp_temp_123456789.py"
        >>> ddp_cleanup(trainer, file)
    """
    if f"{id(trainer)}.py" in file:  # if temp_file suffix in file
        os.remove(file)
