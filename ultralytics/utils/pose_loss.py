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

    def __call__(self, preds, batch):
        """Calculate the total loss and detach it for pose estimation."""
        # Get rank for distributed training (for logging)
        rank = dist.get_rank() if dist.is_initialized() else 0
        gpu_id = self.device.index if hasattr(self.device, 'index') else 0
        log_prefix = f"[Rank {rank}, GPU {gpu_id}] "
        
        # Increment batch counter
        self.batch_counter += 1
        
        loss = torch.zeros(6, device=self.device)  # box, cls, dfl, kpt_location, kpt_visibility, distill
        feats, pred_kpts = preds if isinstance(preds[0], list) else preds[1]
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )

        # B, grids, ..
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_kpts = pred_kpts.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # Targets
        batch_size = pred_scores.shape[0]
        batch_idx = batch["batch_idx"].view(-1, 1)
        targets = torch.cat((batch_idx, batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)
        pred_kpts = self.kpts_decode(anchor_points, pred_kpts.view(batch_size, -1, *self.kpt_shape))  # (b, h*w, 17, 3)

        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[3] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        # Bbox loss
        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[4] = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask
            )
            keypoints = batch["keypoints"].to(self.device).float().clone()
            keypoints[..., 0] *= imgsz[1]
            keypoints[..., 1] *= imgsz[0]

            loss[1], loss[2] = self.calculate_keypoints_loss(
                fg_mask, target_gt_idx, keypoints, batch_idx, stride_tensor, target_bboxes, pred_kpts
            )

        # Calculate distillation loss for the specified target layers if the teacher is available
        should_log = self.batch_counter % self.log_interval == 0  # Only log every log_interval batches
        
        if "teacher" in batch and batch["teacher"] is not None and "teacher_features" in batch and "student_features" in batch:
            # 確保教師模型前向傳播已執行並收集特徵
            teacher = batch["teacher"]
            teacher_features = batch["teacher_features"]
            student_features = batch["student_features"]
            
            # 如果特徵未收集（字典為空），執行前向傳播
            if not teacher_features and teacher is not None:
                if should_log:
                    LOGGER.info(f"{log_prefix}執行教師模型前向傳播以收集特徵...")
                
                with torch.no_grad():
                    try:
                        # 確保設備相符
                        input_device = batch["img"].device
                        # 使用參數獲取教師模型的設備
                        teacher_device = next(teacher.parameters()).device if list(teacher.parameters()) else input_device
                        
                        if teacher_device != input_device:
                            teacher = teacher.to(input_device)
                        
                        # 執行前向傳播
                        _ = teacher(batch["img"])
                        
                        # 更新特徵
                        teacher_features = batch["teacher_features"]
                        
                        if should_log and not teacher_features:
                            LOGGER.warning(f"{log_prefix}教師模型前向傳播後仍無特徵，請檢查勾子註冊")
                    except Exception as e:
                        if should_log:
                            LOGGER.error(f"{log_prefix}教師模型前向傳播錯誤: {str(e)}")
            
            # Check if we have features to compare
            if teacher_features and student_features:
                # Find common keys between teacher and student features
                common_keys = set(teacher_features.keys()) & set(student_features.keys())
                
                # Separate into integer and string keys
                int_keys = sorted([k for k in common_keys if isinstance(k, int)])
                str_keys = sorted([k for k in common_keys if isinstance(k, str)])
                
                # Combine sorted keys
                target_layers = int_keys + str_keys
                
                if not target_layers:
                    if should_log:
                        LOGGER.warning(f"{log_prefix}教師和學生模型沒有共同的特徵層，無法計算蒸餾損失")
                        LOGGER.warning(f"{log_prefix}教師特徵層: {list(teacher_features.keys())}")
                        LOGGER.warning(f"{log_prefix}學生特徵層: {list(student_features.keys())}")
                    loss[5] = torch.tensor(0.0, device=self.device, requires_grad=True)
                else:
                    if should_log:
                        LOGGER.info(f"{log_prefix}計算蒸餾損失，目標層數: {len(target_layers)}, 層索引: {target_layers}")
                    
                    # Calculate distillation loss for each layer
                    distill_losses = []
                    for layer_idx in target_layers:
                        t_feat = teacher_features[layer_idx]
                        s_feat = student_features[layer_idx]
                        
                        # 檢查是否為 None
                        if t_feat is None or s_feat is None:
                            if should_log:
                                LOGGER.warning(f"{log_prefix}層 {layer_idx} 的特徵為 None，跳過")
                            continue
                        
                        # 確保特徵在同一設備上
                        current_device = s_feat.device
                        if t_feat.device != current_device:
                            t_feat = t_feat.to(current_device)
                        
                        # 確保特徵已分離，不會影響教師模型的梯度
                        t_feat = t_feat.detach()
                        
                        # Ensure feature shapes match
                        if t_feat.shape != s_feat.shape:
                            if should_log:
                                LOGGER.warning(f"{log_prefix}層 {layer_idx} 的特徵形狀不匹配: 教師 {t_feat.shape} vs 學生 {s_feat.shape}")
                                
                            # 嘗試調整形狀以匹配
                            try:
                                # 如果教師特徵與學生特徵的通道數不同，可以使用 1x1 卷積進行調整
                                # 這裡我們使用簡單的方法，只保留共同的維度
                                if len(t_feat.shape) == len(s_feat.shape):
                                    # 找出每個維度的最小值
                                    min_dims = [min(td, sd) for td, sd in zip(t_feat.shape, s_feat.shape)]
                                    # 裁剪兩個特徵到相同大小
                                    if len(min_dims) == 4:  # 典型的卷積特徵 [batch, channels, height, width]
                                        t_feat = t_feat[:min_dims[0], :min_dims[1], :min_dims[2], :min_dims[3]]
                                        s_feat = s_feat[:min_dims[0], :min_dims[1], :min_dims[2], :min_dims[3]]
                                        if should_log:
                                            LOGGER.info(f"{log_prefix}調整後的特徵形狀: {t_feat.shape}")
                                    else:
                                        # 其他情況，繼續嘗試下一層
                                        continue
                                else:
                                    # 維度數不同，跳過此層
                                    continue
                            except Exception as e:
                                if should_log:
                                    LOGGER.warning(f"{log_prefix}調整特徵形狀時出錯: {str(e)}")
                                continue
                        
                        # 確保 t_feat 和 s_feat 不是標量
                        if t_feat.numel() == 1 or s_feat.numel() == 1:
                            if should_log:
                                LOGGER.warning(f"{log_prefix}層 {layer_idx} 的特徵是標量，跳過")
                            continue
                        
                        # 計算 MSE 損失
                        try:
                            layer_loss = self.mse_loss(s_feat, t_feat)
                            # 檢查損失是否為 NaN 或無限大
                            if torch.isnan(layer_loss) or torch.isinf(layer_loss):
                                if should_log:
                                    LOGGER.warning(f"{log_prefix}層 {layer_idx} 的蒸餾損失為 NaN 或無限，跳過")
                                continue
                                
                            distill_losses.append(layer_loss)
                            
                            if should_log:
                                LOGGER.info(f"{log_prefix}層 {layer_idx} 的蒸餾損失: {layer_loss.item():.5f}")
                        except Exception as e:
                            if should_log:
                                LOGGER.warning(f"{log_prefix}計算層 {layer_idx} 的蒸餾損失時出錯: {str(e)}")
                            continue
                    
                    if distill_losses:
                        # Combine all layer losses
                        loss[5] = torch.sum(torch.stack(distill_losses))
                        
                        if should_log:
                            LOGGER.info(f"{log_prefix}總蒸餾損失 (未加權): {loss[5].item():.5f}")
                    else:
                        if should_log:
                            LOGGER.warning(f"{log_prefix}沒有計算出有效的蒸餾損失")
                        loss[5] = torch.tensor(0.0, device=self.device, requires_grad=True)
            else:
                # No features collected yet
                if should_log:
                    LOGGER.warning(f"{log_prefix}沒有收集到特徵，無法計算蒸餾損失")
                    LOGGER.warning(f"{log_prefix}教師特徵: {bool(teacher_features)}, 學生特徵: {bool(student_features)}")
                loss[5] = torch.tensor(0.0, device=self.device, requires_grad=True)
        else:
            if should_log:
                missing = []
                if "teacher" not in batch:
                    missing.append("teacher")
                elif batch["teacher"] is None:
                    missing.append("teacher (None)")
                if "teacher_features" not in batch:
                    missing.append("teacher_features")
                if "student_features" not in batch:
                    missing.append("student_features")
                LOGGER.warning(f"{log_prefix}缺少蒸餾所需的組件: {', '.join(missing)}")
            loss[5] = torch.tensor(0.0, device=self.device, requires_grad=True)

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.pose  # pose gain
        loss[2] *= self.hyp.kobj  # kobj gain
        loss[3] *= self.hyp.cls  # cls gain
        loss[4] *= self.hyp.dfl  # dfl gain
        loss[5] *= self.hyp.distill  # distill gain
        
        if should_log:
            # Log all weighted loss components
            LOGGER.info(
                f"{log_prefix}Loss components: box={loss[0]:.4f}, pose={loss[1]:.4f}, "
                f"kobj={loss[2]:.4f}, cls={loss[3]:.4f}, dfl={loss[4]:.4f}, distill={loss[5]:.4f}"
            )

        return loss * batch_size, loss.detach()  # loss(box, cls, dfl)

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
