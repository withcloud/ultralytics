# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import os

from ultralytics.utils import LOGGER
from ultralytics.utils.dev import describe_var
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


class DistillationLoss(nn.Module):
    """Enhanced distillation loss class with multiple loss functions."""
    
    def __init__(self, device):
        """Initialize distillation loss with multiple loss functions."""
        super().__init__()
        self.mse_loss = nn.MSELoss(reduction='mean')
        self.device = device
        
        # 初始化層權重 - 早期層較低，後期層較高
        self.layer_weights = {
            0: 0.6,  # 第一層權重較低
            1: 0.7,
            3: 0.8,
            5: 0.9,
            7: 1.0   # 最後一層權重最高
        }
        
        # 添加字符串形式的層權重對應
        self.layer_name_weights = {
            "model.0.conv": 0.6,  # 第一層權重較低
            "model.1.conv": 0.7,
            "model.3.conv": 0.8,
            "model.5.conv": 0.9,
            "model.7.conv": 1.0   # 最後一層權重最高
        }
    
    def cosine_similarity_loss(self, s_feat, t_feat):
        """Calculate cosine similarity loss between student and teacher features."""
        # 展平特徵
        s_feat_flat = s_feat.view(s_feat.size(0), -1)
        t_feat_flat = t_feat.view(t_feat.size(0), -1)
        
        # 計算餘弦相似度
        cos_sim = F.cosine_similarity(s_feat_flat, t_feat_flat, dim=1)
        # 轉換為損失 (1 - 相似度)
        return (1 - cos_sim).mean()
    
    def attention_loss(self, s_feat, t_feat):
        """Calculate attention-based loss between student and teacher features."""
        # 計算注意力圖 (特徵通道的L2範數)
        s_attention = torch.norm(s_feat, p=2, dim=1)
        t_attention = torch.norm(t_feat, p=2, dim=1)
        
        # L1損失
        return F.l1_loss(s_attention, t_attention)
        
    def statistics_loss(self, s_feat, t_feat):
        """Calculate statistics loss between student and teacher features."""
        # 均值和方差統計
        t_mean = t_feat.mean(dim=[0, 2, 3])
        s_mean = s_feat.mean(dim=[0, 2, 3])
        t_var = t_feat.var(dim=[0, 2, 3])
        s_var = s_feat.var(dim=[0, 2, 3])
        
        # 均值和方差損失
        mean_loss = F.mse_loss(s_mean, t_mean)
        var_loss = F.mse_loss(s_var, t_var)
        
        return (mean_loss + var_loss) * 0.5
    
    def forward(self, teacher_features, student_features, target_layers, temp=1.0):
        """Calculate combined distillation loss with multiple components."""
        distill_losses = []
        
        for layer_idx in target_layers:
            if layer_idx not in teacher_features or layer_idx not in student_features:
                continue
                
            t_feat = teacher_features[layer_idx].detach()
            s_feat = student_features[layer_idx]
            
            # 確保特徵形狀匹配
            if t_feat.shape != s_feat.shape:
                continue
            
            # 獲取層權重，優先使用字符串對應，其次使用數字索引對應，如果都沒有則使用默認值1.0
            if isinstance(layer_idx, str) and layer_idx in self.layer_name_weights:
                weight = self.layer_name_weights[layer_idx]
            elif isinstance(layer_idx, int) and layer_idx in self.layer_weights:
                weight = self.layer_weights[layer_idx]
            else:
                weight = 1.0
            
            # 1. MSE損失
            mse_loss = self.mse_loss(s_feat, t_feat)
            
            # 2. 余弦相似度損失
            cos_loss = self.cosine_similarity_loss(s_feat, t_feat)
            
            # 3. 注意力損失
            att_loss = self.attention_loss(s_feat, t_feat)
            
            # 4. 統計損失
            stat_loss = self.statistics_loss(s_feat, t_feat)
            
            # 組合損失，根據不同層的特性可調整權重
            if (isinstance(layer_idx, int) and layer_idx <= 1) or (isinstance(layer_idx, str) and (layer_idx == "model.0.conv" or layer_idx == "model.1.conv")):
                # 淺層更注重統計信息和注意力
                combined_loss = (mse_loss * 0.4 + cos_loss * 0.2 + 
                                att_loss * 0.2 + stat_loss * 0.2) * weight
            else:
                # 深層更注重MSE和余弦相似度
                combined_loss = (mse_loss * 0.5 + cos_loss * 0.3 + 
                                att_loss * 0.1 + stat_loss * 0.1) * weight
            
            # 打印每層的損失信息，便於調試
            # print(f"Layer {layer_idx}, Weight: {weight:.2f}, MSE: {mse_loss:.4f}, COS: {cos_loss:.4f}, ATT: {att_loss:.4f}, STAT: {stat_loss:.4f}, Combined: {combined_loss:.4f}")
            
            distill_losses.append(combined_loss)
        
        if not distill_losses:
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        
        total_loss = torch.sum(torch.stack(distill_losses))
        # print(f"Total Distillation Loss: {total_loss:.4f} from {len(distill_losses)} layers")
        return total_loss


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
        
        # 使用增强的蒸餾損失
        self.distill_loss = DistillationLoss(self.device)

        self.model = model

        self.teacher_features = {}
        self.student_features = {}
        self.teacher_hooks = []
        self.student_hooks = []

    def __call__(self, preds, batch):
        """Calculate the total loss and detach it for pose estimation."""

        self.teacher_features = {}
        self.student_features = {}

        if "teacher" in batch and batch["teacher"] is not None:
            self.teacher = batch["teacher"].to(self.device)
            self.teacher.eval()
            self.target_layers = batch["target_layers"]

        # 推理之前，要註冊勾子
        # 檢查是否 train_start 為 True
        if "teacher" in batch and batch["teacher"] is not None:
            if batch["train_start"]:
                # 註冊勾子
                print(f"\n\n註冊勾子!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                self.register_teacher_hooks()
                self.register_student_hooks()
        
        # 教師模型推理
        if "teacher" in batch and batch["teacher"] is not None:
            with torch.no_grad():
                teacher_preds = self.teacher(batch["img"])

        # 學生模型推理
        preds = self.model.forward(batch["img"])

        # Get rank for distributed training (for logging)
        rank = dist.get_rank() if dist.is_initialized() else 0
        gpu_id = self.device.index if hasattr(self.device, 'index') else 0
        log_prefix = f"[Rank {rank}, GPU {gpu_id}] "
        
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
        
        if "teacher" in batch and batch["teacher"]:            
            # Check if we have features to compare
            if self.teacher_features and self.student_features:
                # Find common keys between teacher and student features
                common_keys = set(self.teacher_features.keys()) & set(self.student_features.keys())
                
                # Separate into integer and string keys
                int_keys = sorted([k for k in common_keys if isinstance(k, int)])
                str_keys = sorted([k for k in common_keys if isinstance(k, str)])
                
                # Combine sorted keys
                target_layers = int_keys + str_keys
                
                if not target_layers:
                    print(f"{log_prefix}教師和學生模型沒有共同的特徵層，無法計算蒸餾損失")
                    loss[5] = torch.tensor(0.0, device=self.device, requires_grad=True)
                else:
                    # 使用增強的蒸餾損失
                    loss[5] = self.distill_loss(self.teacher_features, self.student_features, target_layers)
            else:
                # No features collected yet
                print(f"{log_prefix}沒有收集到特徵，無法計算蒸餾損失")
                loss[5] = torch.tensor(0.0, device=self.device, requires_grad=True)
        else:
            loss[5] = torch.tensor(0.0, device=self.device, requires_grad=True)

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.pose  # pose gain
        loss[2] *= self.hyp.kobj  # kobj gain
        loss[3] *= self.hyp.cls  # cls gain
        loss[4] *= self.hyp.dfl  # dfl gain
        loss[5] *= self.hyp.distill  # distill gain

        return loss * batch_size, loss.detach()  # loss(box, cls, dfl)
    
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
            
            print(f"{log_prefix}Registering hooks for teacher model:")
            
            # 建立模塊名稱到模塊的映射
            module_dict = {}
            for name, module in self.teacher.named_modules():
                module_dict[name] = module
                
            # 處理每個目標層
            for target in self.target_layers:
                if isinstance(target, int):
                    # 如果是整數索引，直接獲取對應層
                    layer = self.teacher.model[target]
                    layer_full_name = f"model.{target}"
                    layer_idx = target  # 用於hook的layer_idx
                else:
                    # 如果是字符串路徑，從module_dict中查找
                    if target in module_dict:
                        layer = module_dict[target]
                        layer_full_name = target
                        # 對於字符串路徑，我們使用一個唯一標識作為layer_idx
                        layer_idx = target
                    else:
                        print(f"{log_prefix}在教師模型中未找到指定層: {target}")
                        continue
                
                # 獲取層的類型
                layer_type = layer.__class__.__name__
                
                # 構建詳細的層信息
                layer_info = f"層名稱: {layer_full_name}, 類型: {layer_type}"
                
                # 直接檢查該層是否有conv屬性
                if hasattr(layer, 'conv'):
                    layer_info += f", 通道數: {layer.conv.out_channels}"
                else:
                    # 動態查找所有子模塊中的conv
                    conv_modules = []
                    # 獲取層的所有模塊
                    for name, module in layer.named_modules():
                        if hasattr(module, 'conv') and name != '':  # 排除模塊本身
                            if layer_full_name == "model":  # 處理特殊情況
                                full_path = f"{layer_full_name}.{name}.conv"
                            else:
                                full_path = f"{layer_full_name}.{name}.conv" if name else f"{layer_full_name}.conv"
                            conv_info = f"{full_path}: {module.conv.out_channels}通道"
                            conv_modules.append(conv_info)
                    
                    if conv_modules:
                        layer_info += f"\n  子模塊包含:"
                        # 顯示所有conv模塊，但限制數量避免輸出過多
                        max_show = min(len(conv_modules), 5)
                        for j in range(max_show):
                            layer_info += f"\n    - {conv_modules[j]}"
                        if len(conv_modules) > max_show:
                            layer_info += f"\n    ... 等{len(conv_modules)}個conv模塊"
                
                print(f"{log_prefix}{layer_info}")
                
                # 註冊勾子，使用自定義的_save_feature方法並傳遞layer_idx
                self.teacher_hooks.append(layer.register_forward_hook(
                    lambda module, input, output, idx=layer_idx: self._save_teacher_feature(idx, output)))
            
            print(f"{log_prefix}教師模型勾子註冊完成，共 {len(self.teacher_hooks)} 個勾子")

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
        
        print(f"{log_prefix}Registering hooks for student model:")
        
        # 建立模塊名稱到模塊的映射
        module_dict = {}
        for name, module in self.model.named_modules():
            module_dict[name] = module
            
        # 處理每個目標層
        for target in self.target_layers:
            if isinstance(target, int):
                # 如果是整數索引，直接獲取對應層
                layer = self.model.model[target]
                layer_full_name = f"model.{target}"
                layer_idx = target  # 用於hook的layer_idx
            else:
                # 如果是字符串路徑，從module_dict中查找
                if target in module_dict:
                    layer = module_dict[target]
                    layer_full_name = target
                    # 對於字符串路徑，我們使用一個唯一標識作為layer_idx
                    layer_idx = target
                else:
                    print(f"{log_prefix}在學生模型中未找到指定層: {target}")
                    continue
            
            # 獲取層的類型
            layer_type = layer.__class__.__name__
            
            # 構建詳細的層信息
            layer_info = f"層名稱: {layer_full_name}, 類型: {layer_type}"
            
            # 直接檢查該層是否有conv屬性
            if hasattr(layer, 'conv'):
                layer_info += f", 通道數: {layer.conv.out_channels}"
            else:
                # 動態查找所有子模塊中的conv
                conv_modules = []
                # 獲取層的所有模塊
                for name, module in layer.named_modules():
                    if hasattr(module, 'conv') and name != '':  # 排除模塊本身
                        if layer_full_name == "model":  # 處理特殊情況
                            full_path = f"{layer_full_name}.{name}.conv"
                        else:
                            full_path = f"{layer_full_name}.{name}.conv" if name else f"{layer_full_name}.conv"
                        conv_info = f"{full_path}: {module.conv.out_channels}通道"
                        conv_modules.append(conv_info)
                
                if conv_modules:
                    layer_info += f"\n  子模塊包含:"
                    # 顯示所有conv模塊，但限制數量避免輸出過多
                    max_show = min(len(conv_modules), 5)
                    for j in range(max_show):
                        layer_info += f"\n    - {conv_modules[j]}"
                    if len(conv_modules) > max_show:
                        layer_info += f"\n    ... 等{len(conv_modules)}個conv模塊"
            
            print(f"{log_prefix}{layer_info}")
            
            # 註冊勾子
            self.student_hooks.append(layer.register_forward_hook(
                lambda module, input, output, idx=layer_idx: self._save_student_feature(idx, output)))
                
        print(f"{log_prefix}學生模型勾子註冊完成，共 {len(self.student_hooks)} 個勾子")

    def _save_teacher_feature(self, layer_idx, feature):
        """Save features from the teacher model."""
        self.teacher_features[layer_idx] = feature

    def _save_student_feature(self, layer_idx, feature):
        """Save features from the student model."""
        self.student_features[layer_idx] = feature

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
