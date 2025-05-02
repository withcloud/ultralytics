# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import os

from ultralytics.utils import LOGGER
from ultralytics.utils.metrics import OKS_SIGMA
from ultralytics.utils.ops import crop_mask, xywh2xyxy, xyxy2xywh
from ultralytics.utils.tal import RotatedTaskAlignedAssigner, TaskAlignedAssigner, dist2bbox, dist2rbox, make_anchors
from ultralytics.utils.torch_utils import autocast

from .metrics import bbox_iou, probiou
from .tal import bbox2dist
from .loss import v8DetectionLoss

class KeypointLoss(nn.Module):
    """Criterion class for computing keypoint losses."""

    def __init__(self, sigmas) -> None:
        """Initialize the KeypointLoss class with keypoint sigmas."""
        super().__init__()
        self.sigmas = sigmas
        self.keypoint_weights = torch.ones(17)
        
        # 更細緻的權重分配
        self.keypoint_weights[15:17] = 2.5    # 腳踝權重進一步提高
        self.keypoint_weights[13:15] = 2.0    # 膝蓋權重提高
        self.keypoint_weights[11:13] = 1.8    # 髖部權重適度提高
        self.keypoint_weights[5:11] = 1.2     # 軀幹關鍵點適度提高
        self.keypoint_weights[0:5] = 0.8      # 臉部關鍵點降低
        
        # 確保權重平均值為1
        self.keypoint_weights = self.keypoint_weights * (17 / self.keypoint_weights.sum())

    def forward(self, pred_kpts, gt_kpts, kpt_mask, area):
        """Calculate keypoint loss factor and Euclidean distance loss for keypoints."""
        d = (pred_kpts[..., 0] - gt_kpts[..., 0]).pow(2) + (pred_kpts[..., 1] - gt_kpts[..., 1]).pow(2)
        weighted_loss = d * self.keypoint_weights.to(d.device)
        kpt_loss_factor = kpt_mask.shape[1] / (torch.sum(kpt_mask != 0, dim=1) + 1e-9)
        e = weighted_loss / ((2 * self.sigmas).pow(2) * (area + 1e-9) * 2)
        return (kpt_loss_factor.view(-1, 1) * ((1 - torch.exp(-e)) * kpt_mask)).mean()


class v8PoseLoss(v8DetectionLoss):
    """Criterion class for computing training losses for YOLOv8 pose estimation."""

    def __init__(self, model):  # model must be de-paralleled
        """Initialize v8PoseLoss with model parameters and keypoint-specific loss functions."""
        super().__init__(model)
        self.kpt_shape = model.model[-1].kpt_shape
        self.bce_pose = nn.BCEWithLogitsLoss()
        is_pose = self.kpt_shape == [17, 3]
        nkpt = self.kpt_shape[0]  # number of keypoints
        sigmas = torch.from_numpy(OKS_SIGMA).to(self.device) if is_pose else torch.ones(nkpt, device=self.device) / nkpt
        self.keypoint_loss = KeypointLoss(sigmas=sigmas)
        self.mse_loss = nn.MSELoss(reduction='mean')
        
        # Initialize batch counter for logging control
        self.batch_counter = 0
        self.log_interval = 10  # Log every 10 batches

    def __call__(self, preds, batch, teacher=None, distill_factor=0.0, batch_idx=None):
        """Calculate the loss for YOLO predictions and targets."""
        # Initialize loss components (box, cls, dfl)
        loss = torch.zeros(3, device=self.device)
        
        # Extract features from preds
        feats = preds[1] if isinstance(preds, tuple) else preds
        preds = preds[0] if isinstance(preds, tuple) else preds

        # Extract batch information
        if batch_idx is None and 'batch_idx' in batch:
            batch_idx = batch['batch_idx']
        
        # Log info about the batch being processed
        rank = dist.get_rank() if dist.is_initialized() else 0
        gpu_id = self.device.index if hasattr(self.device, 'index') else 0
        log_prefix = f"[Rank {rank}, GPU {gpu_id}] "
        
        if batch_idx is not None:
            batch_info = f"批次 {batch_idx.min().item() if isinstance(batch_idx, torch.Tensor) else batch_idx}"
        else:
            batch_info = "未知批次"
        
        LOGGER.info(f"{log_prefix}{batch_info} - 開始計算loss")
        
        # Check features collection status
        teacher_features_count = 0
        student_features_count = 0
        
        # Extract features from batch if available
        batch_teacher_features = batch.get("teacher_features", {})
        batch_student_features = batch.get("student_features", {})
        
        # 從批次中獲取原始特徵作為備份
        raw_teacher_features = batch.get("raw_teacher_features", {})
        raw_student_features = batch.get("raw_student_features", {})
        
        # Log debug information about features
        LOGGER.info(f"{log_prefix}{batch_info} - 批次中教師特徵: {len(batch_teacher_features)}個, 學生特徵: {len(batch_student_features)}個")
        LOGGER.info(f"{log_prefix}{batch_info} - 原始教師特徵: {len(raw_teacher_features)}個, 原始學生特徵: {len(raw_student_features)}個")
        
        # Calculate the keypoints loss
        keypoints_loss, keypoint_obj_loss = self.calculate_keypoints_loss(preds, batch)
        
        # Calculate object loss
        object_loss = self.calculate_object_loss(preds, batch)
        
        # Add keypoints loss and object loss to total loss
        loss[0] = keypoints_loss
        loss[1] = object_loss
        loss[2] = keypoint_obj_loss
        
        # Calculate distillation loss if teacher is provided and distill_factor > 0
        distill_loss = torch.tensor(0.0, device=self.device)
        if teacher is not None and distill_factor > 0:
            LOGGER.info(f"{log_prefix}{batch_info} - 計算蒸餾loss, 因子: {distill_factor}")
            
            # Check if teacher_features are collected
            if hasattr(teacher, 'teacher_features'):
                teacher_features = teacher.teacher_features
                teacher_features_count = len(teacher_features)
                LOGGER.info(f"{log_prefix}{batch_info} - 教師特徵數量: {teacher_features_count}")
            else:
                teacher_features = {}
                LOGGER.warning(f"{log_prefix}{batch_info} - 教師模型沒有teacher_features屬性")
            
            # Try using features from batch if teacher_features is empty
            if not teacher_features and batch_teacher_features:
                LOGGER.info(f"{log_prefix}{batch_info} - 使用來自batch的教師特徵")
                teacher_features = batch_teacher_features
                teacher_features_count = len(teacher_features)
            
            # 如果仍未獲得教師特徵，嘗試使用原始特徵備份
            if not teacher_features and raw_teacher_features:
                LOGGER.info(f"{log_prefix}{batch_info} - 使用原始教師特徵備份")
                teacher_features = raw_teacher_features
                teacher_features_count = len(teacher_features)
            
            # Check if student_features are collected
            if hasattr(self, 'student_features'):
                student_features = self.student_features
                student_features_count = len(student_features)
                LOGGER.info(f"{log_prefix}{batch_info} - 學生特徵數量: {student_features_count}")
            else:
                student_features = {}
                LOGGER.warning(f"{log_prefix}{batch_info} - loss類沒有student_features屬性")
            
            # Try using features from batch if student_features is empty
            if not student_features and batch_student_features:
                LOGGER.info(f"{log_prefix}{batch_info} - 使用來自batch的學生特徵")
                student_features = batch_student_features
                student_features_count = len(student_features)
            
            # 如果仍未獲得學生特徵，嘗試使用原始特徵備份
            if not student_features and raw_student_features:
                LOGGER.info(f"{log_prefix}{batch_info} - 使用原始學生特徵備份")
                student_features = raw_student_features
                student_features_count = len(student_features)
            
            # Log feature keys for debugging
            if teacher_features:
                LOGGER.info(f"{log_prefix}{batch_info} - 教師特徵鍵: {list(teacher_features.keys())}")
            if student_features:
                LOGGER.info(f"{log_prefix}{batch_info} - 學生特徵鍵: {list(student_features.keys())}")
            
            # If both teacher and student features are collected, compute distillation loss
            if teacher_features_count > 0 and student_features_count > 0:
                try:
                    distill_loss = self.compute_distill_loss(teacher_features, student_features) * distill_factor
                    LOGGER.info(f"{log_prefix}{batch_info} - 蒸餾loss計算成功: {distill_loss.item()}")
                except Exception as e:
                    LOGGER.error(f"{log_prefix}{batch_info} - 計算蒸餾loss時出錯: {str(e)}")
                    distill_loss = torch.tensor(0.0, device=self.device)
            else:
                LOGGER.warning(f"{log_prefix}{batch_info} - 無法計算蒸餾loss: 教師特徵={teacher_features_count}, 學生特徵={student_features_count}")
        
        # Add distillation loss to total loss (if valid)
        if distill_loss.item() != 0.0:
            LOGGER.info(f"{log_prefix}{batch_info} - 添加蒸餾loss到總loss: {distill_loss.item()}")
            # We can add it to any loss component, as it will be weighted by the loss weights later
            loss[2] += distill_loss
        
        # Log final loss components
        LOGGER.info(
            f"{log_prefix}{batch_info} - Loss組成: "
            f"keypoints_loss={loss[0].item():.4f}, "
            f"object_loss={loss[1].item():.4f}, "
            f"keypoint_obj_loss={loss[2].item():.4f}, "
            f"distill_loss={distill_loss.item():.4f}"
        )
        
        # Calculate batch size (number of images in batch)
        batch_size = preds.shape[0]
        
        # Return the loss
        return loss.sum() * batch_size, loss.detach()

    @staticmethod
    def kpts_decode(anchor_points, pred_kpts):
        """Decode predicted keypoints to image coordinates."""
        y = pred_kpts.clone()
        y[..., :2] *= 2.0
        y[..., 0] += anchor_points[:, [0]] - 0.5
        y[..., 1] += anchor_points[:, [1]] - 0.5
        return y

    def calculate_keypoints_loss(self, pred_kpts, batch):
        """Calculate keypoints loss and keypoints object loss for pose estimation.
        
        Args:
            pred_kpts (torch.Tensor): Predicted keypoints features
            batch (dict): Batch data with keypoints and other information
            
        Returns:
            tuple: keypoints localization loss and keypoint visibility loss
        """
        # Get device and shape info from pred_kpts
        device = pred_kpts.device
        bs, nk, h, w = pred_kpts.shape  # bs: batch size, nk: num keypoints, h: height, w: width
        
        # Initialize loss tensors
        keypoints_loss = torch.tensor(0.0, device=device)
        keypoint_obj_loss = torch.tensor(0.0, device=device)
        
        # Get keypoints from batch
        if "keypoints" not in batch:
            LOGGER.warning(f"批次數據中沒有找到keypoints，返回零損失")
            return keypoints_loss, keypoint_obj_loss
        
        keypoints = batch["keypoints"]
        
        # Get indices info for batch processing
        batch_idx = batch["batch_idx"] if "batch_idx" in batch else torch.zeros(keypoints.shape[0], device=device)
        num_kpts = keypoints.shape[1]  # Number of keypoints per instance
        num_classes = self.nc  # Number of classes
        
        # Get keypoints from prediction
        pred_kpts = pred_kpts.permute(0, 2, 3, 1)  # [bs, h, w, nk]
        
        # Scale the keypoint coordinates to match heatmap dimensions
        scale_x = w
        scale_y = h
        
        # Initialize counters for normalization
        valid_samples = 0
        
        # Iterate over each sample in the batch
        for bi in range(bs):
            # Get batch indices for current sample
            b_idx = (batch_idx == bi).nonzero().squeeze(1)
            if len(b_idx) == 0:
                continue  # No objects in this batch item
            
            # Get keypoints for current sample
            kpts = keypoints[b_idx]  # [num_instances, num_kpts, 3]
            
            # Check if keypoints are valid
            if kpts.shape[0] == 0 or kpts.numel() == 0:
                continue
            
            # Generate target heatmaps
            target_heatmaps = self.generate_kpt_heatmaps(kpts, h, w)  # [num_instances, num_kpts, h, w]
            
            # Compute MSE loss for keypoint localization
            for i in range(kpts.shape[0]):  # For each instance
                for k in range(num_kpts):  # For each keypoint
                    # Check if keypoint is visible
                    if kpts[i, k, 2] > 0:
                        # Compute MSE between predicted heatmap and target Gaussian heatmap
                        pred_heatmap = pred_kpts[bi, :, :, k]
                        target_heatmap = target_heatmaps[i, k]
                        
                        # Calculate MSE loss
                        mse_loss = F.mse_loss(pred_heatmap, target_heatmap)
                        keypoints_loss += mse_loss
                        
                        # For visibility (obj prediction)
                        if nk // num_kpts > 1:  # If we have obj prediction channel
                            # Assume last channel is for visibility
                            vis_idx = num_kpts + k
                            pred_vis = pred_kpts[bi, :, :, vis_idx]
                            
                            # Create target visibility mask (1 where keypoint is, 0 elsewhere)
                            vis_target = torch.zeros_like(target_heatmap)
                            # Mark keypoint location as 1 for visibility
                            x, y = torch.round(kpts[i, k, 0] * scale_x).long(), torch.round(kpts[i, k, 1] * scale_y).long()
                            if 0 <= x < w and 0 <= y < h:
                                vis_target[y, x] = 1.0
                            
                            # Binary cross-entropy for visibility
                            bce_loss = F.binary_cross_entropy_with_logits(pred_vis, vis_target)
                            keypoint_obj_loss += bce_loss
                
                valid_samples += 1
        
        # Average the loss over valid samples
        if valid_samples > 0:
            keypoints_loss = keypoints_loss / valid_samples
            keypoint_obj_loss = keypoint_obj_loss / valid_samples
        
        # Apply weights
        keypoints_loss = keypoints_loss * self.keypoint_weight
        keypoint_obj_loss = keypoint_obj_loss * self.obj_weight
        
        return keypoints_loss, keypoint_obj_loss
    
    def generate_kpt_heatmaps(self, keypoints, height, width, sigma=2.0):
        """Generate Gaussian heatmaps for the keypoints.
        
        Args:
            keypoints (torch.Tensor): Keypoints tensor of shape [num_instances, num_kpts, 3]
            height (int): Height of the heatmap
            width (int): Width of the heatmap
            sigma (float): Standard deviation for Gaussian kernel
            
        Returns:
            torch.Tensor: Heatmaps of shape [num_instances, num_kpts, height, width]
        """
        num_instances = keypoints.shape[0]
        num_kpts = keypoints.shape[1]
        
        # Initialize heatmaps
        heatmaps = torch.zeros((num_instances, num_kpts, height, width), device=keypoints.device)
        
        # Create coordinate grid
        y_grid, x_grid = torch.meshgrid(torch.arange(height, device=keypoints.device), 
                                        torch.arange(width, device=keypoints.device))
        
        # Parameters for the Gaussian distribution
        size = 6 * sigma + 3
        x = torch.arange(0, size, 1, dtype=torch.float, device=keypoints.device)
        y = x[:, None]
        x0, y0 = size // 2, size // 2
        g = torch.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
        
        # Iterate over instances and keypoints to compute the heatmap
        for i in range(num_instances):
            for k in range(num_kpts):
                # Check if keypoint is visible
                if keypoints[i, k, 2] > 0:
                    # Get keypoint coordinates (normalized)
                    x, y = keypoints[i, k, 0], keypoints[i, k, 1]
                    
                    # Scale to heatmap dimensions
                    x = x * width
                    y = y * height
                    
                    # Convert to integers
                    xi, yi = int(x), int(y)
                    
                    # Check if keypoint is within bounds
                    if 0 <= xi < width and 0 <= yi < height:
                        # Compute the Gaussian distribution
                        left, right = min(xi, size // 2), min(width - xi, size // 2 + 1)
                        top, bottom = min(yi, size // 2), min(height - yi, size // 2 + 1)
                        
                        # Create a patch for the Gaussian
                        masked_grid_x = x_grid[yi - top:yi + bottom, xi - left:xi + right]
                        masked_grid_y = y_grid[yi - top:yi + bottom, xi - left:xi + right]
                        
                        # Create the Gaussian heatmap for this keypoint
                        masked_gaussian = g[size // 2 - top:size // 2 + bottom, size // 2 - left:size // 2 + right]
                        
                        # Update the heatmap
                        heatmaps[i, k, yi - top:yi + bottom, xi - left:xi + right] = masked_gaussian
        
        return heatmaps
    
    def calculate_object_loss(self, preds, batch):
        """Calculate the object (bbox) loss for pose estimation.
        
        Args:
            preds (torch.Tensor): Predicted features
            batch (dict): Batch data with boxes and other information
            
        Returns:
            torch.Tensor: object loss
        """
        # Initialize loss tensor
        obj_loss = torch.tensor(0.0, device=self.device)
        
        # Check if we have bboxes in the batch
        if "bboxes" not in batch:
            # For pose estimation, we might not need bbox loss if we're focusing on keypoints
            return obj_loss
        
        # Extract bboxes from batch
        bboxes = batch["bboxes"]
        
        # Check if bboxes are valid
        if bboxes.shape[0] == 0:
            return obj_loss
        
        # Extract batch indices
        batch_idx = batch["batch_idx"] if "batch_idx" in batch else torch.zeros(bboxes.shape[0], device=self.device)
        
        # Assume preds contains object confidence predictions
        # This would depend on your specific model architecture
        # For now, we return a placeholder zero loss
        # In a real implementation, you would compare predicted bboxes with ground truth
        
        return obj_loss

    def compute_distill_loss(self, teacher_features, student_features):
        """計算教師模型和學生模型之間的特徵蒸餾損失。
        
        Args:
            teacher_features (dict): 教師模型特徵字典，鍵為層名稱，值為特徵張量
            student_features (dict): 學生模型特徵字典，鍵為層名稱，值為特徵張量
            
        Returns:
            torch.Tensor: 蒸餾損失值
        """
        # 獲取當前設備和日誌前綴
        device = self.device
        rank = dist.get_rank() if dist.is_initialized() else 0
        gpu_id = device.index if hasattr(device, 'index') else device
        log_prefix = f"[Rank {rank}, GPU {gpu_id}] "
        
        # 初始化蒸餾損失
        total_distill_loss = torch.tensor(0.0, device=device)
        
        # 檢查特徵字典是否為空
        if not teacher_features:
            LOGGER.warning(f"{log_prefix}教師特徵字典為空，返回零損失")
            return total_distill_loss
            
        if not student_features:
            LOGGER.warning(f"{log_prefix}學生特徵字典為空，返回零損失")
            return total_distill_loss
        
        # 記錄特徵鍵和數量
        teacher_keys = list(teacher_features.keys())
        student_keys = list(student_features.keys())
        LOGGER.info(f"{log_prefix}教師特徵鍵({len(teacher_keys)}): {teacher_keys}")
        LOGGER.info(f"{log_prefix}學生特徵鍵({len(student_keys)}): {student_keys}")
        
        # 嘗試匹配教師和學生特徵
        try:
            # 匹配特徵 - 基於模型結構配對
            matched_features = self._match_features_by_structure(teacher_features, student_features)
            
            if not matched_features:
                LOGGER.warning(f"{log_prefix}無法基於結構匹配特徵，嘗試基於鍵匹配")
                # 嘗試直接匹配相同的鍵
                matched_features = []
                for t_key in teacher_keys:
                    if t_key in student_keys:
                        matched_features.append((t_key, t_key))
            
            # 記錄匹配結果
            LOGGER.info(f"{log_prefix}匹配到 {len(matched_features)} 對特徵")
            for t_key, s_key in matched_features:
                LOGGER.info(f"{log_prefix}匹配: 教師 '{t_key}' -> 學生 '{s_key}'")
            
            # 如果仍然沒有匹配的特徵，返回零損失
            if not matched_features:
                LOGGER.warning(f"{log_prefix}無法匹配任何特徵，返回零損失")
                return total_distill_loss
                
            # 計算每對匹配特徵的損失
            feature_count = 0
            for t_key, s_key in matched_features:
                t_feat = teacher_features[t_key]
                s_feat = student_features[s_key]
                
                # 驗證特徵是否為張量
                if not isinstance(t_feat, torch.Tensor) or not isinstance(s_feat, torch.Tensor):
                    LOGGER.warning(f"{log_prefix}特徵不是張量: 教師 '{t_key}' 或 學生 '{s_key}'")
                    continue
                    
                # 記錄特徵形狀和設備
                LOGGER.info(f"{log_prefix}特徵 {feature_count + 1}: 教師 '{t_key}' {t_feat.shape} @ {t_feat.device}, 學生 '{s_key}' {s_feat.shape} @ {s_feat.device}")
                
                # 確保特徵在同一設備上
                if t_feat.device != s_feat.device:
                    LOGGER.info(f"{log_prefix}將教師特徵從 {t_feat.device} 移動到 {s_feat.device}")
                    t_feat = t_feat.to(s_feat.device)
                
                # 嘗試匹配形狀差異
                try:
                    # 僅在需要時適應特徵
                    if t_feat.shape != s_feat.shape:
                        LOGGER.info(f"{log_prefix}特徵形狀不匹配，嘗試調整: {t_feat.shape} vs {s_feat.shape}")
                        
                        # 檢查是否可以通過平均池化匹配
                        if t_feat.dim() >= 3 and s_feat.dim() >= 3:
                            # 獲取空間維度（通常是最後兩個）
                            t_spatial = t_feat.shape[-2:]
                            s_spatial = s_feat.shape[-2:]
                            
                            # 檢查是否需要空間調整
                            if t_spatial != s_spatial:
                                LOGGER.info(f"{log_prefix}應用自適應池化匹配空間維度: {t_spatial} -> {s_spatial}")
                                # 使用自適應池化調整空間維度
                                adaptive_pool = nn.AdaptiveAvgPool2d(s_spatial)
                                t_feat = adaptive_pool(t_feat)
                            
                            # 檢查通道維度是否相同
                            if t_feat.shape[1] != s_feat.shape[1]:
                                LOGGER.warning(f"{log_prefix}通道數不匹配且無法自動調整: {t_feat.shape[1]} vs {s_feat.shape[1]}")
                                continue
                        else:
                            LOGGER.warning(f"{log_prefix}特徵維度不支持自動調整: 教師 {t_feat.dim()}D vs 學生 {s_feat.dim()}D")
                            continue
                    
                    # 最終檢查形狀是否匹配
                    if t_feat.shape != s_feat.shape:
                        LOGGER.warning(f"{log_prefix}調整後特徵形狀仍不匹配: {t_feat.shape} vs {s_feat.shape}, 跳過此特徵")
                        continue
                        
                    # 計算MSE損失
                    feature_loss = F.mse_loss(s_feat, t_feat.detach())
                    
                    # 檢查損失值是否有效
                    if torch.isnan(feature_loss) or torch.isinf(feature_loss):
                        LOGGER.warning(f"{log_prefix}特徵損失值無效: {feature_loss}")
                        continue
                        
                    LOGGER.info(f"{log_prefix}特徵 {feature_count + 1} 損失: {feature_loss.item():.6f}")
                    
                    # 累加損失
                    total_distill_loss += feature_loss
                    feature_count += 1
                    
                except Exception as e:
                    LOGGER.error(f"{log_prefix}計算特徵損失時出錯: {str(e)}")
                    import traceback
                    LOGGER.debug(traceback.format_exc())
                    continue
            
            # 平均所有特徵的損失
            if feature_count > 0:
                total_distill_loss = total_distill_loss / feature_count
                LOGGER.info(f"{log_prefix}平均蒸餾損失: {total_distill_loss.item():.6f}")
            else:
                LOGGER.warning(f"{log_prefix}沒有成功計算任何特徵損失，返回零損失")
                total_distill_loss = torch.tensor(0.0, device=device)
                
        except Exception as e:
            LOGGER.error(f"{log_prefix}計算蒸餾損失時發生異常: {str(e)}")
            import traceback
            LOGGER.debug(traceback.format_exc())
            total_distill_loss = torch.tensor(0.0, device=device)
            
        return total_distill_loss
    
    def _match_features_by_structure(self, teacher_features, student_features):
        """嘗試基於模型結構匹配教師和學生特徵。
        
        Args:
            teacher_features (dict): 教師模型特徵字典
            student_features (dict): 學生模型特徵字典
            
        Returns:
            list: 匹配的(教師特徵鍵, 學生特徵鍵)列表
        """
        # 初始化結果列表
        matched_pairs = []
        
        # 解析層名稱和索引
        teacher_layers = {}
        student_layers = {}
        
        # 解析教師層
        for key in teacher_features.keys():
            parts = key.split('.')
            if len(parts) >= 2 and parts[0] == 'model':
                try:
                    layer_idx = int(parts[1])
                    teacher_layers[layer_idx] = key
                except ValueError:
                    continue
        
        # 解析學生層
        for key in student_features.keys():
            parts = key.split('.')
            if len(parts) >= 2 and parts[0] == 'model':
                try:
                    layer_idx = int(parts[1])
                    student_layers[layer_idx] = key
                except ValueError:
                    continue
        
        # 直接匹配相同索引的層
        teacher_indices = sorted(teacher_layers.keys())
        student_indices = sorted(student_layers.keys())
        
        # 沒有層索引信息時，嘗試直接匹配相同的鍵
        if not teacher_indices or not student_indices:
            # 檢查是否有完全相同的鍵
            common_keys = set(teacher_features.keys()) & set(student_features.keys())
            for key in common_keys:
                matched_pairs.append((key, key))
            return matched_pairs
            
        # 記錄索引信息
        log_prefix = f"[特徵匹配] "
        LOGGER.info(f"{log_prefix}教師層索引: {teacher_indices}")
        LOGGER.info(f"{log_prefix}學生層索引: {student_indices}")
        
        # 嘗試層級匹配
        if len(teacher_indices) == len(student_indices):
            # 如果層數量相同，直接一一對應
            for t_idx, s_idx in zip(teacher_indices, student_indices):
                matched_pairs.append((teacher_layers[t_idx], student_layers[s_idx]))
        else:
            # 如果層數量不同，嘗試更複雜的匹配邏輯
            # 例如，可以基於相對位置進行匹配
            t_max = max(teacher_indices)
            s_max = max(student_indices)
            
            for t_idx in teacher_indices:
                # 使用相對位置找到最接近的學生層
                relative_pos = t_idx / t_max
                closest_s_idx = min(student_indices, key=lambda s: abs(s / s_max - relative_pos))
                matched_pairs.append((teacher_layers[t_idx], student_layers[closest_s_idx]))
                
        return matched_pairs

# Helper class to extract intermediate features from a layer
class ExtractLayerHook:
    def __init__(self, module):
        self.features = None
        self.hook = module.register_forward_hook(self.hook_fn)
        
    def hook_fn(self, module, input, output):
        self.features = output
        
    def remove(self):
        self.hook.remove()
