import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
import numpy as np

# GIoU Loss
def giou_loss(pred_boxes, target_boxes):
    pred_x1 = pred_boxes[..., 0] - pred_boxes[..., 2] / 2
    pred_y1 = pred_boxes[..., 1] - pred_boxes[..., 3] / 2
    pred_x2 = pred_boxes[..., 0] + pred_boxes[..., 2] / 2
    pred_y2 = pred_boxes[..., 1] + pred_boxes[..., 3] / 2

    target_x1 = target_boxes[..., 0] - target_boxes[..., 2] / 2
    target_y1 = target_boxes[..., 1] - target_boxes[..., 3] / 2
    target_x2 = target_boxes[..., 0] + target_boxes[..., 2] / 2
    target_y2 = target_boxes[..., 1] + target_boxes[..., 3] / 2

    x1 = torch.max(pred_x1, target_x1)
    y1 = torch.max(pred_y1, target_y1)
    x2 = torch.min(pred_x2, target_x2)
    y2 = torch.min(pred_y2, target_y2)

    inter_area = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
    pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
    target_area = (target_x2 - target_x1) * (target_y2 - target_y1)
    union_area = pred_area + target_area - inter_area

    iou = inter_area / (union_area + 1e-6)

    enc_x1 = torch.min(pred_x1, target_x1)
    enc_y1 = torch.min(pred_y1, target_y1)
    enc_x2 = torch.max(pred_x2, target_x2)
    enc_y2 = torch.max(pred_y2, target_y2)
    enc_area = (enc_x2 - enc_x1) * (enc_y2 - enc_y1)

    giou = iou - (enc_area - union_area) / (enc_area + 1e-6)
    return 1 - giou

# Focal Loss
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return F_loss.mean()

# Hungarian Matching for DETR
def hungarian_matcher(pred_boxes, pred_logits, target_boxes, target_labels, num_classes):
    B = pred_boxes.shape[0]
    indices = []
    for i in range(B):
        if len(target_boxes[i]) == 0:
            indices.append((torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long)))
            continue
        valid_labels = target_labels[i][target_labels[i] < num_classes]
        if len(valid_labels) == 0:
            indices.append((torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long)))
            continue
        out_prob = pred_logits[i].softmax(-1)
        cost_class = -out_prob[:, valid_labels]
        cost_bbox = torch.cdist(pred_boxes[i], target_boxes[i][:len(valid_labels)], p=1)
        cost_giou = giou_loss(pred_boxes[i].unsqueeze(1), target_boxes[i][:len(valid_labels)].unsqueeze(0)).squeeze(0)
        C = cost_bbox + 0.5 * cost_giou + cost_class
        C = C.cpu().detach().numpy()
        pred_idx, tgt_idx = linear_sum_assignment(C)
        indices.append((torch.tensor(pred_idx, dtype=torch.long), torch.tensor(tgt_idx, dtype=torch.long)))
    return indices