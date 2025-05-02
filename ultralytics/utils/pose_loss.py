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

    def calculate_keypoints_loss(
        self, masks, target_gt_idx, keypoints, batch_idx, stride_tensor, target_bboxes, pred_kpts
    ):
        """
        Calculate the keypoints loss for the model.

        This function calculates the keypoints loss and keypoints object loss for a given batch. The keypoints loss is
        based on the difference between the predicted keypoints and ground truth keypoints. The keypoints object loss is
        a binary classification loss that classifies whether a keypoint is present or not.

        Args:
            masks (torch.Tensor): Binary mask tensor indicating object presence, shape (BS, N_anchors).
            target_gt_idx (torch.Tensor): Index tensor mapping anchors to ground truth objects, shape (BS, N_anchors).
            keypoints (torch.Tensor): Ground truth keypoints, shape (N_kpts_in_batch, N_kpts_per_object, kpts_dim).
            batch_idx (torch.Tensor): Batch index tensor for keypoints, shape (N_kpts_in_batch, 1).
            stride_tensor (torch.Tensor): Stride tensor for anchors, shape (N_anchors, 1).
            target_bboxes (torch.Tensor): Ground truth boxes in (x1, y1, x2, y2) format, shape (BS, N_anchors, 4).
            pred_kpts (torch.Tensor): Predicted keypoints, shape (BS, N_anchors, N_kpts_per_object, kpts_dim).

        Returns:
            kpts_loss (torch.Tensor): The keypoints loss.
            kpts_obj_loss (torch.Tensor): The keypoints object loss.
        """
        batch_idx = batch_idx.flatten()
        batch_size = len(masks)

        # Find the maximum number of keypoints in a single image
        max_kpts = torch.unique(batch_idx, return_counts=True)[1].max()

        # Create a tensor to hold batched keypoints
        batched_keypoints = torch.zeros(
            (batch_size, max_kpts, keypoints.shape[1], keypoints.shape[2]), device=keypoints.device
        )

        # TODO: any idea how to vectorize this?
        # Fill batched_keypoints with keypoints based on batch_idx
        for i in range(batch_size):
            keypoints_i = keypoints[batch_idx == i]
            batched_keypoints[i, : keypoints_i.shape[0]] = keypoints_i

        # Expand dimensions of target_gt_idx to match the shape of batched_keypoints
        target_gt_idx_expanded = target_gt_idx.unsqueeze(-1).unsqueeze(-1)

        # Use target_gt_idx_expanded to select keypoints from batched_keypoints
        selected_keypoints = batched_keypoints.gather(
            1, target_gt_idx_expanded.expand(-1, -1, keypoints.shape[1], keypoints.shape[2])
        )

        # Divide coordinates by stride
        selected_keypoints[..., :2] /= stride_tensor.view(1, -1, 1, 1)

        kpts_loss = 0
        kpts_obj_loss = 0

        if masks.any():
            gt_kpt = selected_keypoints[masks]
            area = xyxy2xywh(target_bboxes[masks])[:, 2:].prod(1, keepdim=True)
            pred_kpt = pred_kpts[masks]
            
            # 修改這裡：將閾值設為 0.3
            kpt_mask = gt_kpt[..., 2] > 0.3 if gt_kpt.shape[-1] == 3 else torch.full_like(gt_kpt[..., 0], True)

            kpts_loss = self.keypoint_loss(pred_kpt, gt_kpt, kpt_mask, area)  # pose loss

            if pred_kpt.shape[-1] == 3:
                kpts_obj_loss = self.bce_pose(pred_kpt[..., 2], kpt_mask.float())  # keypoint obj loss

        return kpts_loss, kpts_obj_loss

# Helper class to extract intermediate features from a layer
class ExtractLayerHook:
    def __init__(self, module):
        self.features = None
        self.hook = module.register_forward_hook(self.hook_fn)
        
    def hook_fn(self, module, input, output):
        self.features = output
        
    def remove(self):
        self.hook.remove()
