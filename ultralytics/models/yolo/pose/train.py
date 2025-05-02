# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from copy import copy
import os

from ultralytics.models import yolo
from ultralytics.nn.tasks import PoseModel
from ultralytics.utils import DEFAULT_CFG, LOGGER, callbacks
from ultralytics.utils.plotting import plot_images, plot_results
import torch
import torch.distributed as dist
from ultralytics.nn.tasks import attempt_load_weights
import torch.nn.functional as F


class PoseTrainer(yolo.detect.DetectionTrainer):
    """
    A class extending the DetectionTrainer class for training YOLO pose estimation models.

    This trainer specializes in handling pose estimation tasks, managing model training, validation, and visualization
    of pose keypoints alongside bounding boxes.

    Attributes:
        args (dict): Configuration arguments for training.
        model (PoseModel): The pose estimation model being trained.
        data (dict): Dataset configuration including keypoint shape information.
        loss_names (Tuple[str]): Names of the loss components used in training.

    Methods:
        get_model: Retrieves a pose estimation model with specified configuration.
        set_model_attributes: Sets keypoints shape attribute on the model.
        get_validator: Creates a validator instance for model evaluation.
        plot_training_samples: Visualizes training samples with keypoints.
        plot_metrics: Generates and saves training/validation metric plots.

    Examples:
        >>> from ultralytics.models.yolo.pose import PoseTrainer
        >>> args = dict(model="yolo11n-pose.pt", data="coco8-pose.yaml", epochs=3)
        >>> trainer = PoseTrainer(overrides=args)
        >>> trainer.train()
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """
        Initialize a PoseTrainer object for training YOLO pose estimation models.

        This initializes a trainer specialized for pose estimation tasks, setting the task to 'pose' and
        handling specific configurations needed for keypoint detection models.

        Args:
            cfg (dict, optional): Default configuration dictionary containing training parameters.
            overrides (dict, optional): Dictionary of parameter overrides for the default configuration.
                Supported distillation parameters:
                - teacher: Path to teacher model weights file
                - distill: Weight for distillation loss (default: 1.0)
                - freezeAllBN: Whether to freeze all BatchNorm layers (default: False)
                - target_layers: List of layer indices or names for feature distillation (default: ["model.6", "model.8", "model.10"])
            _callbacks (list, optional): List of callback functions to be executed during training.

        Notes:
            This trainer will automatically set the task to 'pose' regardless of what is provided in overrides.
            A warning is issued when using Apple MPS device due to known bugs with pose models.

        Examples:
            >>> from ultralytics.models.yolo.pose import PoseTrainer
            >>> args = dict(model="yolov8n-pose.pt", data="coco8-pose.yaml", epochs=3)
            >>> trainer = PoseTrainer(overrides=args)
            >>> trainer.train()
        """
        if overrides is None:
            overrides = {}
        overrides["task"] = "pose"

        self.teacher_path = overrides.get("teacher", None)  # Store the teacher path instead of model
        self.teacher = None  # Initialized to None, will load the model later
        self.distill = overrides.get("distill", 1.0)
        self.freezeAllBN = overrides.get("freezeAllBN", False)
        
        # 輸出更詳細的初始化信息
        rank = 0  # 在初始化階段還沒有 dist.get_rank()
        if 'RANK' in os.environ:
            rank = int(os.environ['RANK'])
        log_prefix = f"[Rank {rank}] "
        
        # 新的默認目標層 - 使用早期的層以確保更容易匹配
        default_layers = ["model.0", "model.1", "model.2"]  # 使用模型前面的層，這些層通常存在於所有模型中
        
        # 從配置中獲取目標層
        self.target_layers = overrides.get("target_layers", default_layers)
        
        # 輸出目標層信息
        LOGGER.info(f"{log_prefix}使用以下目標層進行蒸餾: {self.target_layers}")
        
        if not self.target_layers and self.teacher_path:
            LOGGER.warning(f"{log_prefix}未指定目標層，使用默認值: {default_layers}，可通過 'target_layers' 參數指定")
            self.target_layers = default_layers
        
        # 將索引轉換為字符串路徑，避免 DDP 環境中的問題
        for i, layer in enumerate(self.target_layers):
            if isinstance(layer, int):
                self.target_layers[i] = f"model.{layer}"
                LOGGER.info(f"{log_prefix}將索引 {layer} 轉換為路徑 'model.{layer}'")
        
        # For collecting features from layers
        self.teacher_features = {}
        self.student_features = {}
        self.teacher_hooks = []
        self.student_hooks = []

        # Initialize parent class first to set up device and other attributes
        super().__init__(cfg, overrides, _callbacks)

        # Now we can initialize the teacher model since self.device is available
        if self.teacher_path is not None:
            LOGGER.info(f"{log_prefix}將初始化教師模型: {self.teacher_path}")
            self.init_teacher_model()
            
            if _callbacks is None:
                _callbacks = callbacks.get_default_callbacks()
            
            # 確保所有必要的回調都被註冊
            _callbacks["on_train_start"].append(self.on_train_start)
            _callbacks["on_train_epoch_start"].append(self.on_epoch_start)
            _callbacks["on_train_epoch_end"].append(self.on_epoch_end)
            _callbacks["on_val_start"].append(self.on_val_start)
            _callbacks["on_val_end"].append(self.on_val_end)
            _callbacks["on_train_end"].append(self.on_train_end)
            _callbacks["teardown"].append(self.teardown)
            _callbacks["on_batch_end"].append(self.on_batch_end)
            _callbacks["on_train_batch_start"].append(self.on_train_batch_start)  # 新添加的批次開始回調

        if isinstance(self.args.device, str) and self.args.device.lower() == "mps":
            LOGGER.warning(
                "Apple MPS known Pose bug. Recommend 'device=cpu' for Pose models. "
                "See https://github.com/ultralytics/ultralytics/issues/4031."
            )

    def init_teacher_model(self):
        """Initialize the teacher model on the current device."""
        if self.teacher_path is not None and self.teacher is None:
            # Get rank for distributed training
            rank = dist.get_rank() if dist.is_initialized() else 0
            gpu_id = self.device.index if hasattr(self.device, 'index') else 0
            
            # Log detailed information about which GPU is loading the teacher
            log_prefix = f"[Rank {rank}, GPU {gpu_id}] "
            LOGGER.info(f"{log_prefix}Loading teacher model from {self.teacher_path}")
            
            # Load teacher model using attempt_load_weights to avoid circular imports
            try:
                if hasattr(torch.cuda, 'memory_allocated'):
                    mem_before = torch.cuda.memory_allocated(self.device) / (1024 ** 2)  # MB
                
                # Load the model using attempt_load_weights instead of YOLO
                self.teacher = attempt_load_weights(self.teacher_path, device=self.device)
                
                # Freeze teacher parameters
                for k, v in self.teacher.named_parameters():
                    v.requires_grad = False

                # Set teacher model to eval mode
                self.teacher.eval()

                # Freeze BN layers
                for m in self.teacher.modules():
                    if isinstance(m, (torch.nn.BatchNorm2d, torch.nn.BatchNorm1d)):
                        m.eval()  # Only set BN layers to eval mode
                        for param in m.parameters():
                            param.requires_grad = False
                
                if hasattr(torch.cuda, 'memory_allocated'):
                    mem_after = torch.cuda.memory_allocated(self.device) / (1024 ** 2)  # MB
                    mem_used = mem_after - mem_before
                    LOGGER.info(f"{log_prefix}Teacher model loaded successfully. Memory used: {mem_used:.2f} MB")
                else:
                    LOGGER.info(f"{log_prefix}Teacher model loaded successfully.")
                
                # Log teacher model structure details
                teacher_params = sum(p.numel() for p in self.teacher.parameters())
                LOGGER.info(f"{log_prefix}Teacher model has {teacher_params:,} parameters")
                
                # Log process ID for debugging
                LOGGER.info(f"{log_prefix}Process ID: {os.getpid()}")
                
                # 確保立即註冊 hook
                self.register_teacher_hooks()
                
            except Exception as e:
                LOGGER.error(f"{log_prefix}Error loading teacher model: {str(e)}")
                raise
    
    def ensure_features_collection(self, batch):
        """Ensure features are collected by running forward pass if needed."""
        # Get rank for distributed training
        rank = dist.get_rank() if dist.is_initialized() else 0
        gpu_id = self.device.index if hasattr(self.device, 'index') else 0
        log_prefix = f"[Rank {rank}, GPU {gpu_id}] "
        
        # Clear features explicitly
        self.teacher_features = {}
        self.student_features = {}
        
        # Add teacher to batch if not present
        if self.teacher is not None and "teacher" not in batch:
            batch["teacher"] = self.teacher
            
        # Add feature containers to batch
        batch["teacher_features"] = self.teacher_features
        batch["student_features"] = self.student_features
        
        # Get image device
        input_device = batch["img"].device
        
        # 確保教師模型在前向傳播前已經被註冊了勾子
        if len(self.teacher_hooks) == 0 and self.teacher is not None:
            LOGGER.warning(f"{log_prefix}教師模型沒有註冊勾子，正在重新註冊...")
            self.register_teacher_hooks()
            
        # 確保學生模型在前向傳播前已經被註冊了勾子
        if len(self.student_hooks) == 0:
            LOGGER.warning(f"{log_prefix}學生模型沒有註冊勾子，正在重新註冊...")
            self.register_student_hooks()
        
        LOGGER.info(f"{log_prefix}當前教師模型勾子數: {len(self.teacher_hooks)}, 學生模型勾子數: {len(self.student_hooks)}")
        
        # 強制執行教師模型前向傳播以收集特徵 (明確指定)
        if self.teacher is not None:
            try:
                LOGGER.info(f"{log_prefix}執行教師模型前向傳播以收集特徵...")
                
                # 確保教師模型處於評估模式
                self.teacher.eval()
                
                # 確保教師模型在正確的設備上
                teacher_device = None
                try:
                    # 嘗試獲取模型設備
                    if isinstance(self.teacher, torch.nn.parallel.DistributedDataParallel):
                        teacher_device = next(self.teacher.module.parameters()).device
                    else:
                        teacher_device = next(self.teacher.parameters()).device
                    
                    if teacher_device != input_device:
                        LOGGER.info(f"{log_prefix}教師模型設備 ({teacher_device}) 與輸入設備 ({input_device}) 不匹配，正在移動模型...")
                        if isinstance(self.teacher, torch.nn.parallel.DistributedDataParallel):
                            # 對於 DDP 模型，我們不能直接移動模型
                            LOGGER.warning(f"{log_prefix}教師模型是 DDP 模型，無法直接移動。請確保模型與輸入在相同設備上。")
                        else:
                            self.teacher = self.teacher.to(input_device)
                            LOGGER.info(f"{log_prefix}教師模型已移至 {input_device}")
                except Exception as e:
                    LOGGER.warning(f"{log_prefix}獲取教師模型設備時發生錯誤: {str(e)}")
                
                # 執行前向傳播 (不計算梯度)
                with torch.no_grad():
                    _ = self.teacher(batch["img"])
                
                # 檢查是否收集到了特徵
                if not self.teacher_features:
                    LOGGER.warning(f"{log_prefix}教師模型前向傳播後仍無特徵，請檢查勾子註冊")
                    
                    # 檢查模型結構
                    if isinstance(self.teacher, torch.nn.parallel.DistributedDataParallel):
                        LOGGER.info(f"{log_prefix}使用目標層: {self.target_layers}")
                        LOGGER.info(f"{log_prefix}模型類型: {type(self.teacher).__name__}, 模塊類型: {type(self.teacher.module).__name__}")
                    else:
                        LOGGER.info(f"{log_prefix}使用目標層: {self.target_layers}")
                        LOGGER.info(f"{log_prefix}模型類型: {type(self.teacher).__name__}")
            except Exception as e:
                LOGGER.error(f"{log_prefix}執行教師模型前向傳播時發生錯誤: {str(e)}")
        
        # 將教師特徵更新到批次中
        batch["teacher_features"] = self.teacher_features
        batch["student_features"] = self.student_features
        
        return batch
            
    def preprocess_batch(self, batch):
        """處理每個批次數據，添加教師模型和特徵到批次中"""
        batch = super().preprocess_batch(batch)
        
        # 確保特徵被正確收集
        batch = self.ensure_features_collection(batch)
        
        return batch

    def _find_layer(self, model, layer_id):
        """查找模型中的指定層"""
        # 獲取進程和設備信息
        rank = dist.get_rank() if dist.is_initialized() else 0
        gpu_id = self.device.index if hasattr(self.device, 'index') else 0
        log_prefix = f"[Rank {rank}, GPU {gpu_id}] "
        
        # 檢查是否為 DDP 模型
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            LOGGER.info(f"{log_prefix}在 DDP 模型中查找層: {layer_id}")
            base_model = model.module
        else:
            base_model = model
        
        # 嘗試不同的方式查找層
        layer = None
        error_messages = []
        
        # 1. 直接使用整數索引
        if isinstance(layer_id, int):
            try:
                layer = base_model.model[layer_id]
                LOGGER.info(f"{log_prefix}使用整數索引 {layer_id} 找到層: {type(layer).__name__}")
                return layer
            except (IndexError, AttributeError) as e:
                error_messages.append(f"使用整數索引失敗: {str(e)}")
        
        # 2. 使用字符串路徑 (例如 "model.0")
        elif isinstance(layer_id, str):
            # 移除 "model." 前綴以獲取索引
            if layer_id.startswith("model."):
                try:
                    idx = int(layer_id.split(".")[1])
                    if "conv" in layer_id:  # 處理 "model.0.conv" 格式
                        try:
                            layer = base_model.model[idx].conv
                            LOGGER.info(f"{log_prefix}使用路徑 {layer_id} 找到層: {type(layer).__name__}")
                            return layer
                        except (IndexError, AttributeError) as e:
                            error_messages.append(f"訪問 .conv 失敗: {str(e)}")
                    else:  # 處理 "model.0" 格式
                        try:
                            layer = base_model.model[idx]
                            LOGGER.info(f"{log_prefix}使用路徑 {layer_id} 找到層: {type(layer).__name__}")
                            return layer
                        except (IndexError, AttributeError) as e:
                            error_messages.append(f"訪問索引失敗: {str(e)}")
                except (ValueError, IndexError) as e:
                    error_messages.append(f"解析索引失敗: {str(e)}")
        
        # 3. 嘗試使用 named_modules 查找
        try:
            for name, module in base_model.named_modules():
                if name == layer_id or (layer_id.isdigit() and name == f"model.{layer_id}"):
                    layer = module
                    LOGGER.info(f"{log_prefix}使用 named_modules 找到層 {layer_id}: {type(layer).__name__}")
                    return layer
        except Exception as e:
            error_messages.append(f"使用 named_modules 失敗: {str(e)}")
        
        # 4. 最後嘗試: 列出可用層
        try:
            available_layers = []
            for i, m in enumerate(base_model.model):
                available_layers.append(f"model.{i} ({type(m).__name__})")
                if hasattr(m, 'conv'):
                    available_layers.append(f"model.{i}.conv ({type(m.conv).__name__})")
            
            LOGGER.warning(f"{log_prefix}無法找到層 {layer_id}")
            LOGGER.info(f"{log_prefix}可用層: {available_layers[:10]}...")
            if len(available_layers) > 10:
                LOGGER.info(f"{log_prefix}...以及其他 {len(available_layers)-10} 個層")
            
            # 提供建議的目標層
            suggested_layers = [f"model.{i}" for i in range(min(10, len(list(base_model.model))))]
            LOGGER.info(f"{log_prefix}建議使用以下層: {suggested_layers}")
            
        except Exception as e:
            error_messages.append(f"列出可用層失敗: {str(e)}")
            LOGGER.error(f"{log_prefix}所有嘗試均失敗: {error_messages}")
        
        return None  # 無法找到層
    
    def register_teacher_hooks(self):
        """Register hooks on the teacher model to capture intermediate features."""
        if self.teacher is not None:
            # Clear any existing hooks
            for hook in self.teacher_hooks:
                hook.remove()
            self.teacher_hooks = []
            
            # Get rank for distributed training
            rank = dist.get_rank() if dist.is_initialized() else 0
            gpu_id = self.device.index if hasattr(self.device, 'index') else 0
            log_prefix = f"[Rank {rank}, GPU {gpu_id}] "
            
            LOGGER.info(f"{log_prefix}Registering hooks for teacher model:")
            
            # 檢查是否為 DDP 模型
            if isinstance(self.teacher, torch.nn.parallel.DistributedDataParallel):
                LOGGER.info(f"{log_prefix}檢測到教師模型為 DDP 模型，使用 teacher.module")
                base_teacher = self.teacher.module
            else:
                base_teacher = self.teacher
            
            # 輸出教師模型結構
            try:
                model_structure = []
                for i, layer in enumerate(base_teacher.model):
                    model_structure.append(f"model.{i}: {type(layer).__name__}")
                LOGGER.info(f"{log_prefix}教師模型結構: {model_structure[:5]}...")
                if len(model_structure) > 5:
                    LOGGER.info(f"{log_prefix}...以及其他 {len(model_structure)-5} 個層")
            except Exception as e:
                LOGGER.warning(f"{log_prefix}無法獲取教師模型結構: {str(e)}")
                
            # 處理每個目標層
            for target in self.target_layers:
                # 查找層
                layer = self._find_layer(base_teacher, target)
                
                if layer is None:
                    LOGGER.warning(f"{log_prefix}無法找到教師模型層: {target}，跳過")
                    continue
                
                layer_idx = target  # 用於 hook 的 layer_idx
                
                # 註冊勾子，使用自定義的_save_feature方法並傳遞layer_idx
                hook = layer.register_forward_hook(
                    lambda module, input, output, idx=layer_idx: self._save_teacher_feature(idx, output))
                self.teacher_hooks.append(hook)
                LOGGER.info(f"{log_prefix}為教師模型層 {target} 註冊了勾子")
            
            # 在模型的前向傳播開始之前註冊一個鉤子，用於清空特徵
            def pre_forward_hook(module, input):
                self.teacher_features = {}
                return None
            
            if isinstance(self.teacher, torch.nn.parallel.DistributedDataParallel):
                pre_hook = self.teacher.module.register_forward_pre_hook(pre_forward_hook)
            else:
                pre_hook = self.teacher.register_forward_pre_hook(pre_forward_hook)
            
            self.teacher_hooks.append(pre_hook)
            
            if not self.teacher_hooks:
                LOGGER.warning(f"{log_prefix}沒有為教師模型註冊任何勾子！")
            else:
                LOGGER.info(f"{log_prefix}教師模型勾子註冊完成，共 {len(self.teacher_hooks)} 個勾子")
    
    def register_student_hooks(self):
        """Register hooks on the student model to capture intermediate features."""
        # Clear any existing hooks
        for hook in self.student_hooks:
            hook.remove()
        self.student_hooks = []
        
        # Get rank for distributed training
        rank = dist.get_rank() if dist.is_initialized() else 0
        gpu_id = self.device.index if hasattr(self.device, 'index') else 0
        log_prefix = f"[Rank {rank}, GPU {gpu_id}] "
        
        LOGGER.info(f"{log_prefix}Registering hooks for student model:")
        
        # 檢查是否為 DDP 模型
        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            LOGGER.info(f"{log_prefix}檢測到 DDP 模型，使用 model.module 而不是 model")
            base_model = self.model.module
        else:
            base_model = self.model
            
        # 輸出學生模型結構
        try:
            model_structure = []
            for i, layer in enumerate(base_model.model):
                model_structure.append(f"model.{i}: {type(layer).__name__}")
            LOGGER.info(f"{log_prefix}學生模型結構: {model_structure[:5]}...")
            if len(model_structure) > 5:
                LOGGER.info(f"{log_prefix}...以及其他 {len(model_structure)-5} 個層")
        except Exception as e:
            LOGGER.warning(f"{log_prefix}無法獲取學生模型結構: {str(e)}")
            
        # 處理每個目標層
        for target in self.target_layers:
            # 查找層
            layer = self._find_layer(base_model, target)
            
            if layer is None:
                LOGGER.warning(f"{log_prefix}無法找到學生模型層: {target}，跳過")
                continue
            
            layer_idx = target  # 用於 hook 的 layer_idx
            
            # 註冊勾子
            hook = layer.register_forward_hook(
                lambda module, input, output, idx=layer_idx: self._save_student_feature(idx, output))
            self.student_hooks.append(hook)
            LOGGER.info(f"{log_prefix}為學生模型層 {target} 註冊了勾子")
                
        # 在模型的前向傳播開始之前註冊一個鉤子，用於清空特徵
        def pre_forward_hook(module, input):
            self.student_features = {}
            return None
        
        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            pre_hook = self.model.module.register_forward_pre_hook(pre_forward_hook)
        else:
            pre_hook = self.model.register_forward_pre_hook(pre_forward_hook)
        
        self.student_hooks.append(pre_hook)
        
        if not self.student_hooks:
            LOGGER.warning(f"{log_prefix}沒有為學生模型註冊任何勾子！")
        else:
            LOGGER.info(f"{log_prefix}學生模型勾子註冊完成，共 {len(self.student_hooks)} 個勾子")

    def _save_teacher_feature(self, layer_idx, feature):
        """Save features from the teacher model."""
        self.teacher_features[layer_idx] = feature
        
    def _save_student_feature(self, layer_idx, feature):
        """Save features from the student model."""
        self.student_features[layer_idx] = feature

    def _model_train(self):
        """Set model in training mode."""
        self.model.train()
        # Freeze BN stat
        for n, m in self.model.named_modules():
            if any(filter(lambda f: f in n, self.freeze_layer_names)) and isinstance(m, torch.nn.BatchNorm2d):
                m.eval()

        # 凍結BN層，讓它們的統計數據(running_mean, running_var)不會更新
        if self.freezeAllBN:
            for m in self.model.modules():
                if isinstance(m, (torch.nn.BatchNorm2d, torch.nn.BatchNorm1d)):
                    m.eval()  # 只有BN層設為評估模式
                    for param in m.parameters():
                        param.requires_grad = False

    def on_train_start(self, trainer):
        # Get rank for distributed training
        rank = dist.get_rank() if dist.is_initialized() else 0
        gpu_id = self.device.index if hasattr(self.device, 'index') else 0
        log_prefix = f"[Rank {rank}, GPU {gpu_id}] "
        
        LOGGER.info(f"{log_prefix}Starting training...")
        
        if self.teacher is not None:
            # 打印教師模型和學生模型的結構
            LOGGER.info(f"{log_prefix}" + "=" * 80)
            LOGGER.info(f"{log_prefix}教師模型結構:")
            for name, module in self.teacher.named_modules():
                if name.startswith("model.") and len(name.split(".")) <= 3:
                    module_type = module.__class__.__name__
                    num_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
                    has_conv = hasattr(module, 'conv')
                    channels_info = f", 通道數: {module.conv.out_channels}" if has_conv else ""
                    LOGGER.info(f"{log_prefix}  - {name}: {module_type} (參數量: {num_params}){channels_info}")
            
            LOGGER.info(f"\n{log_prefix}學生模型結構:")
            for name, module in self.model.named_modules():
                if name.startswith("model.") and len(name.split(".")) <= 3:
                    module_type = module.__class__.__name__
                    num_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
                    has_conv = hasattr(module, 'conv')
                    channels_info = f", 通道數: {module.conv.out_channels}" if has_conv else ""
                    LOGGER.info(f"{log_prefix}  - {name}: {module_type} (參數量: {num_params}){channels_info}")
            LOGGER.info(f"{log_prefix}" + "=" * 80)

            # Register hooks for the teacher model
            self.register_teacher_hooks()
            # Register hooks for the student model
            self.register_student_hooks()

    def on_epoch_start(self, trainer):
        # Get rank for distributed training
        rank = dist.get_rank() if dist.is_initialized() else 0
        gpu_id = self.device.index if hasattr(self.device, 'index') else 0
        log_prefix = f"[Rank {rank}, GPU {gpu_id}] "
        
        LOGGER.info(f"{log_prefix}Starting epoch {trainer.epoch}/{trainer.epochs}")
        
        self.model.epoch = trainer.epoch
        self.model.epochs = trainer.epochs
        self.model.is_first_batch_in_epoch = True

    def on_epoch_end(self, trainer):
        # Get rank for distributed training
        rank = dist.get_rank() if dist.is_initialized() else 0
        gpu_id = self.device.index if hasattr(self.device, 'index') else 0
        log_prefix = f"[Rank {rank}, GPU {gpu_id}] "
        
        LOGGER.info(f"{log_prefix}Finished epoch {trainer.epoch}/{trainer.epochs}")

    def on_val_start(self, trainer):
        # Get rank for distributed training
        rank = dist.get_rank() if dist.is_initialized() else 0
        gpu_id = self.device.index if hasattr(self.device, 'index') else 0
        log_prefix = f"[Rank {rank}, GPU {gpu_id}] "
        
        LOGGER.info(f"{log_prefix}Starting validation...")

    def on_val_end(self, trainer):
        # Get rank for distributed training
        rank = dist.get_rank() if dist.is_initialized() else 0
        gpu_id = self.device.index if hasattr(self.device, 'index') else 0
        log_prefix = f"[Rank {rank}, GPU {gpu_id}] "
        
        LOGGER.info(f"{log_prefix}Validation completed")
    
    def on_train_end(self, trainer):
        # Get rank for distributed training
        rank = dist.get_rank() if dist.is_initialized() else 0
        gpu_id = self.device.index if hasattr(self.device, 'index') else 0
        log_prefix = f"[Rank {rank}, GPU {gpu_id}] "
        
        LOGGER.info(f"{log_prefix}Training completed, cleaning up...")
        
        # Remove hooks when training ends
        for hook in self.teacher_hooks:
            hook.remove()
        for hook in self.student_hooks:
            hook.remove()

        # Clear the stored features
        self.teacher_features = {}
        self.student_features = {}
        
        LOGGER.info(f"{log_prefix}Cleanup completed")
    
    def teardown(self, trainer):
        # Get rank for distributed training
        rank = dist.get_rank() if dist.is_initialized() else 0
        gpu_id = self.device.index if hasattr(self.device, 'index') else 0
        log_prefix = f"[Rank {rank}, GPU {gpu_id}] "
        
        LOGGER.info(f"{log_prefix}Teardown in progress...")
        
        # Make sure all hooks are removed
        for hook in self.teacher_hooks:
            hook.remove()
        for hook in self.student_hooks:
            hook.remove()
            
        LOGGER.info(f"{log_prefix}Teardown completed")
    
    def on_batch_end(self, trainer):
        self.model.is_first_batch_in_epoch = False
        pass
        
    def on_train_batch_start(self, trainer, batch, batch_idx):
        """Callback triggered at the start of the batch process."""
        # Get rank for distributed training
        rank = dist.get_rank() if dist.is_initialized() else 0
        gpu_id = self.device.index if hasattr(self.device, 'index') else 0
        log_prefix = f"[Rank {rank}, GPU {gpu_id}] "
        
        # 只在第一批或每 10 個批次執行特徵收集和打印一次
        if (trainer.epoch == 0 and batch_idx == 0) or (trainer.epoch % 10 == 0 and batch_idx % 10 == 0):
            LOGGER.info(f"{log_prefix}當前教師模型勾子數: {len(self.teacher_hooks)}, 學生模型勾子數: {len(self.student_hooks)}")
        
        # 當模型是 DistributedDataParallel 時，獲取當前批次的索引
        bi = batch_idx

        # 確保有足夠的勾子
        if self.teacher is not None and len(self.teacher_hooks) < len(self.target_layers):
            self.register_teacher_hooks()
        if len(self.student_hooks) < len(self.target_layers):
            self.register_student_hooks()
        
        # 步驟 1: 收集模型特徵
        batch = self.ensure_features_collection(batch)
        
        # 步驟 2: 計算蒸餾損失
        if self.teacher is not None:
            d_loss, batch = self.calculate_distillation_loss(batch)
            
            # Add distillation loss to the batch for total loss computation in model's loss method
            if not torch.isnan(d_loss) and not torch.isinf(d_loss) and d_loss > 0:
                batch['d_loss'] = d_loss.item()  # For logging
                batch['d_loss_tensor'] = d_loss  # The actual tensor for loss computation
            else:
                batch['d_loss'] = 0.0  # For logging
                batch['d_loss_tensor'] = torch.tensor(0.0, device=self.device)  # Dummy tensor
        
        return batch

    def set_target_layers(self, new_target_layers):
        """
        設置新的目標層並重新註冊勾子。
        
        Args:
            new_target_layers (list): 包含層索引或層名稱的列表，例如 [0, 6, 13] 或 ["model.0.conv", "model.6.cv1"]
        """
        # Get rank for distributed training
        rank = dist.get_rank() if dist.is_initialized() else 0
        gpu_id = self.device.index if hasattr(self.device, 'index') else 0
        log_prefix = f"[Rank {rank}, GPU {gpu_id}] "
        
        # 更新目標層
        self.target_layers = new_target_layers
        LOGGER.info(f"{log_prefix}更新目標層為: {self.target_layers}")
        
        # 重新註冊勾子
        self.register_teacher_hooks()
        self.register_student_hooks()
        
        return self.target_layers

    def get_model(self, cfg=None, weights=None, verbose=True):
        """
        Get pose estimation model with specified configuration and weights.

        Args:
            cfg (str | Path | dict | None): Model configuration file path or dictionary.
            weights (str | Path | None): Path to the model weights file.
            verbose (bool): Whether to display model information.

        Returns:
            (PoseModel): Initialized pose estimation model.
        """
        model = PoseModel(
            cfg, nc=self.data["nc"], ch=self.data["channels"], data_kpt_shape=self.data["kpt_shape"], verbose=verbose
        )
        if weights:
            model.load(weights)

        return model

    def set_model_attributes(self):
        """Sets keypoints shape attribute of PoseModel."""
        super().set_model_attributes()
        self.model.kpt_shape = self.data["kpt_shape"]

    def get_validator(self):
        """Returns an instance of the PoseValidator class for validation."""
        self.loss_names = "box_loss", "pose_loss", "kobj_loss", "cls_loss", "dfl_loss", "d_loss"
        return yolo.pose.PoseValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )

    def plot_training_samples(self, batch, ni):
        """
        Plot a batch of training samples with annotated class labels, bounding boxes, and keypoints.

        Args:
            batch (dict): Dictionary containing batch data with the following keys:
                - img (torch.Tensor): Batch of images
                - keypoints (torch.Tensor): Keypoints coordinates for pose estimation
                - cls (torch.Tensor): Class labels
                - bboxes (torch.Tensor): Bounding box coordinates
                - im_file (list): List of image file paths
                - batch_idx (torch.Tensor): Batch indices for each instance
            ni (int): Current training iteration number used for filename

        The function saves the plotted batch as an image in the trainer's save directory with the filename
        'train_batch{ni}.jpg', where ni is the iteration number.
        """
        images = batch["img"]
        kpts = batch["keypoints"]
        cls = batch["cls"].squeeze(-1)
        bboxes = batch["bboxes"]
        paths = batch["im_file"]
        batch_idx = batch["batch_idx"]
        plot_images(
            images,
            batch_idx,
            cls,
            bboxes,
            kpts=kpts,
            paths=paths,
            fname=self.save_dir / f"train_batch{ni}.jpg",
            on_plot=self.on_plot,
        )

    def plot_metrics(self):
        """Plots training/val metrics."""
        plot_results(file=self.csv, pose=True, on_plot=self.on_plot)  # save results.png

    def calculate_distillation_loss(self, batch):
        """Calculate distillation loss between teacher and student features."""
        # Get rank for distributed training
        rank = dist.get_rank() if dist.is_initialized() else 0
        gpu_id = self.device.index if hasattr(self.device, 'index') else 0
        log_prefix = f"[Rank {rank}, GPU {gpu_id}] "
        
        # Check if we have the necessary components
        if (not self.teacher_features or not self.student_features) and self.teacher is not None:
            LOGGER.warning(f"{log_prefix}沒有收集到特徵，無法計算蒸餾損失")
            LOGGER.info(f"{log_prefix}教師特徵: {bool(self.teacher_features)}, 學生特徵: {bool(self.student_features)}")
            
            # Try to ensure features are collected again
            batch = self.ensure_features_collection(batch)
            
            # Check again after collection attempt
            if not self.teacher_features or not self.student_features:
                missing_components = []
                if not hasattr(batch, 'teacher') or batch['teacher'] is None:
                    missing_components.append("teacher")
                if not self.teacher_features:
                    missing_components.append("teacher_features")
                if not self.student_features:
                    missing_components.append("student_features")
                
                LOGGER.warning(f"{log_prefix}缺少蒸餾所需的組件: {', '.join(missing_components)}")
                return 0.0, batch
        
        d_loss = 0.0
        
        try:
            # Loop through all features that should be matching
            for layer_idx in self.target_layers:
                # 確保層索引是字符串形式，以支持 DDP 環境
                if isinstance(layer_idx, int):
                    layer_idx = f"model.{layer_idx}"
                
                # 檢查是否有收集到相應層的特徵
                if layer_idx not in self.teacher_features or layer_idx not in self.student_features:
                    if layer_idx not in self.teacher_features:
                        LOGGER.warning(f"{log_prefix}教師模型沒有層 {layer_idx} 的特徵")
                    if layer_idx not in self.student_features:
                        LOGGER.warning(f"{log_prefix}學生模型沒有層 {layer_idx} 的特徵")
                    continue
                
                # Get features from teacher and student
                t_feat = self.teacher_features[layer_idx]
                s_feat = self.student_features[layer_idx]
                
                # Ensure they're on the same device
                if t_feat.device != s_feat.device:
                    LOGGER.info(f"{log_prefix}特徵設備不匹配: 教師({t_feat.device}) vs 學生({s_feat.device})，正在調整...")
                    t_feat = t_feat.to(s_feat.device)
                
                # Ensure shapes match or can be adapted
                if t_feat.shape != s_feat.shape:
                    # 如果通道數不同，則使用自適應平均池化進行調整
                    if t_feat.shape[1] != s_feat.shape[1]:  # 通道數不同
                        LOGGER.info(
                            f"{log_prefix}特徵通道數不匹配 (層 {layer_idx}): "
                            f"教師({t_feat.shape[1]}) vs 學生({s_feat.shape[1]})，將使用 MSE 損失"
                        )
                    else:  # 空間維度不同
                        # 將教師特徵調整為與學生特徵相同的空間尺寸
                        if len(t_feat.shape) == 4:  # For 2D features
                            t_feat = F.interpolate(t_feat, size=s_feat.shape[2:], mode='bilinear', align_corners=False)
                            LOGGER.info(
                                f"{log_prefix}調整教師特徵大小: {t_feat.shape[2:]} -> {s_feat.shape[2:]} (層 {layer_idx})"
                            )
                
                # Calculate MSE loss between teacher and student features
                mse_loss = F.mse_loss(s_feat, t_feat)
                d_loss += mse_loss
                
                # Log the individual loss components
                LOGGER.debug(f"{log_prefix}層 {layer_idx} 的蒸餾損失: {mse_loss.item():.6f}")
            
            # Apply distillation weight
            d_loss *= self.distill
            
            # Add to batch for later processing
            batch['d_loss'] = d_loss.item()
            
            LOGGER.info(f"{log_prefix}總蒸餾損失 (d_loss): {d_loss.item():.6f}")
            
        except Exception as e:
            LOGGER.error(f"{log_prefix}計算蒸餾損失時發生錯誤: {str(e)}")
            import traceback
            LOGGER.error(traceback.format_exc())
            d_loss = 0.0
            batch['d_loss'] = 0.0
        
        return d_loss, batch
