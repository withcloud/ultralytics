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
        """Calculate the total loss and detach to CPU."""
        loss = torch.zeros(6, device=self.device)  # box, pose, cls, dfl, kobj_loss, d_loss
        pred_kpts = torch.empty(0, device=self.device)
        
        # 檢查是否存在蒸餾損失
        d_loss_tensor = batch.get('d_loss_tensor', None)
        
        if not isinstance(preds, list):  # for Training (vs. Validation)
            feats, pred_kpts = preds if len(preds) == 2 else (preds, None)
            pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
                (self.reg_max * 4, self.nc), 1)
            
            # b: batch size, grids: number of grid cells (h*w), ..., c: class preds, g: normalized offsets,
            # e.g. n: number of kpts, i.e. 2 * nk (nk: number of keypoints)
            batch_size, channels, h, w = feats[0].shape  # batch size, channels, height, width
            
            # targets: normalized, cls_id, x, y, w, h, kp, kp, ...
            pred_bboxes = self.bbox_decode(pred_distri)  # [b, h*w, 4], xyxy (+/- regression max)
            targets = torch.cat((batch["cls"].view(-1, 1), batch["bboxes"]), 1)
            imgs = batch["img"]
            
            # Index into batch data using masks
            if self.use_kobj_aug > 0 and hasattr(batch, 'mask_kobj'):
                # Create a mask for the original data (not the keypoint-occluded object data)
                mask_orig = ~batch['mask_kobj'].view(-1)
                
                # Apply the mask to relevant tensors to select only the original data
                targets_orig = targets[mask_orig]
                imgs_orig = imgs[mask_orig]
            else:
                targets_orig = targets
                imgs_orig = imgs
            
            mask_gt = targets_orig.sum(1) > 0  # Remove empty targets
            targets_orig = targets_orig[mask_gt]
            indices = batch["batch_idx"].view(-1, 1).repeat(1, targets_orig.shape[1])[mask_gt]
            
            # Build targets
            _, targets_masks, ang_targets, gt_kpts = self.build_targets(
                (batch_size, channels, h, w), pred_bboxes, targets_orig, indices, imgs_orig)
                
            # Compute losses
            (
                box_loss,
                pose_loss,
                cls_loss,
                dfl_loss,
                kobj_loss,
            ) = self.compute_loss(
                pred_scores, pred_distri, pred_bboxes, pred_kpts, targets_masks, gt_kpts, ang_targets, imgs.shape[2:4]
            )
            
            # Combine box, cls, dfl losses before multiplying with their respective loss weights
            loss[0] = box_loss * self.box_weight  # box loss
            loss[1] = pose_loss * self.pose_weight  # pose loss
            loss[2] = cls_loss * self.cls_weight  # cls loss
            loss[3] = dfl_loss * self.dfl_weight  # dfl loss
            loss[4] = kobj_loss * self.kobj_weight  # kobj loss
            
            # 添加蒸餾損失
            if d_loss_tensor is not None and d_loss_tensor.item() > 0:
                if isinstance(d_loss_tensor, torch.Tensor):
                    loss[5] = d_loss_tensor  # d_loss
                else:
                    try:
                        # 如果是數值而不是張量，則轉換為張量
                        loss[5] = torch.tensor(d_loss_tensor, device=self.device)
                    except Exception as e:
                        LOGGER.error(f"無法將 d_loss_tensor 轉換為張量: {e}")
                        loss[5] = torch.tensor(0.0, device=self.device)
            else:
                loss[5] = torch.tensor(0.0, device=self.device)
                
            # 記錄損失組件
            rank = dist.get_rank() if dist.is_initialized() else 0
            LOGGER.info(f"[Rank {rank}, GPU {self.device.index if hasattr(self.device, 'index') else 0}] Loss components: box={loss[0].item():.4f}, pose={loss[1].item():.4f}, kobj={loss[4].item():.4f}, cls={loss[2].item():.4f}, dfl={loss[3].item():.4f}, distill={loss[5].item():.4f}")
        else:  # for Validation (vs. Training)
            loss[0] = loss[1] = loss[2] = loss[3] = loss[4] = loss[5] = torch.tensor(0.0, device=self.device)
            pred_kpts = preds[1]
            
        # Final loss = 損失結合   
        return loss.sum(), loss.detach(), pred_kpts

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
