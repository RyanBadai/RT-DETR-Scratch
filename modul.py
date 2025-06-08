import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm
import time
from torchvision.ops import nms
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import supervision as sv

#  ResNet-like Backbone
class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, 128, 2, stride=2)
        self.layer2 = self._make_layer(128, 256, 2, stride=2)
        self.layer3 = self._make_layer(256, 512, 2, stride=2)
        self.layer4 = self._make_layer(512, 1024, 2, stride=2)

        # Add 1x1 convolutions to adjust channel sizes
        self.adjust_s4 = nn.Conv2d(256, 128, kernel_size=1)
        self.adjust_s5 = nn.Conv2d(512, 256, kernel_size=1)
        self.adjust_s6 = nn.Conv2d(1024, 512, kernel_size=1)


    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(1, blocks):
            layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        s3 = self.layer1(x)
        s4 = self.layer2(s3)
        s5 = self.layer3(s4)
        s6 = self.layer4(s5)

        # Adjust channel sizes
        s4 = self.adjust_s4(s4)
        s5 = self.adjust_s5(s5)
        s6 = self.adjust_s6(s6)

        return [s4, s5, s6]

# Attention-based Intra-scale Feature Interaction (AIFI)
class AIFI(nn.Module):
    def __init__(self, dim):
        super(AIFI, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=dim, num_heads=8, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.ReLU(),
            nn.Linear(dim * 4, dim)
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        B, C, H, W = x.shape
        seq_len = H * W
        x = x.view(B, C, seq_len).permute(0, 2, 1)
        attn_output, _ = self.attention(x, x, x)
        x = self.norm(x + attn_output)
        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)
        x = x.permute(0, 2, 1).view(B, C, H, W)
        return x

# Cross-scale Feature-fusion Module (CCFM)
class CCFM(nn.Module):
    def __init__(self, channels):
        super(CCFM, self).__init__()
        self.conv_s4 = nn.Conv2d(channels[0], channels[2], kernel_size=1)
        self.conv_s5 = nn.Conv2d(channels[1], channels[2], kernel_size=1)
        self.fusion = nn.Conv2d(channels[2] * 3, channels[2], kernel_size=3, padding=1)
        self.norm = nn.BatchNorm2d(channels[2])
        self.relu = nn.ReLU(inplace=True)

    def forward(self, s4, s5, s6):
        s4 = F.interpolate(self.conv_s4(s4), size=s6.shape[2:], mode='bilinear', align_corners=False)
        s5 = F.interpolate(self.conv_s5(s5), size=s6.shape[2:], mode='bilinear', align_corners=False)
        fused = torch.cat([s4, s5, s6], dim=1)
        fused = self.fusion(fused)
        fused = self.norm(fused)
        fused = self.relu(fused)
        return fused

# Efficient Hybrid Encoder
class EfficientHybridEncoder(nn.Module):
    def __init__(self, channels=[128, 256, 512]):
        super(EfficientHybridEncoder, self).__init__()
        self.aifi = AIFI(channels[2])
        self.ccfm = CCFM(channels)

    def forward(self, features):
        s4, s5, s6 = features
        fused = self.ccfm(s4, s5, s6)
        output = self.aifi(fused)
        return output

# IoU-aware Query Selection
class IoUQuerySelection(nn.Module):
    def __init__(self, dim, num_queries=300):
        super(IoUQuerySelection, self).__init__()
        self.query_embed = nn.Embedding(num_queries, dim)
        self.fc = nn.Linear(dim, dim)

    def forward(self, x):
        B, C, _, _ = x.shape
        queries = self.query_embed.weight.unsqueeze(0).repeat(B, 1, 1)
        queries = self.fc(queries)
        return queries

# RT-DETR Decoder
class RTDETRDecoder(nn.Module):
    def __init__(self, dim, num_queries=300, num_classes=3):
        super(RTDETRDecoder, self).__init__()
        self.layers = nn.ModuleList([
            nn.TransformerDecoderLayer(d_model=dim, nhead=8, batch_first=True) for _ in range(6)
        ])
        self.query_selection = IoUQuerySelection(dim, num_queries)
        self.fc_bbox = nn.Linear(dim, 4)
        self.fc_cls = nn.Linear(dim, num_classes)

    def forward(self, memory):
        B, C, H, W = memory.shape
        queries = self.query_selection(memory)
        memory = memory.view(B, C, H * W).permute(0, 2, 1)
        for layer in self.layers:
            queries = layer(queries, memory)
        bboxes = self.fc_bbox(queries).sigmoid()
        cls_scores = self.fc_cls(queries)
        return bboxes, cls_scores

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

# RT-DETR Model
class RTDETR(nn.Module):
    def __init__(self, num_classes=3):
        super(RTDETR, self).__init__()
        self.backbone = ResNet()
        self.encoder = EfficientHybridEncoder()
        self.decoder = RTDETRDecoder(dim=512, num_classes=num_classes)

    def forward(self, x):
        features = self.backbone(x)
        memory = self.encoder(features)
        bboxes, cls_scores = self.decoder(memory)
        return bboxes, cls_scores

# Compute IoU for Precision/Recall
def compute_iou(box1, box2):
    x1, y1, w1, h1 = box1[..., 0], box1[..., 1], box1[..., 2], box1[..., 3]
    x2, y2, w2, h2 = box2[..., 0], box2[..., 1], box2[..., 2], box2[..., 3]

    x1_min = x1 - w1 / 2
    y1_min = y1 - h1 / 2
    x1_max = x1 + w1 / 2
    y1_max = y1 + h1 / 2

    x2_min = x2 - w2 / 2
    y2_min = y2 - h2 / 2
    x2_max = x2 + w2 / 2
    y2_max = y2 + h2 / 2

    inter_x_min = torch.max(x1_min, x2_min)
    inter_y_min = torch.max(y1_min, y2_min)
    inter_x_max = torch.min(x1_max, x2_max)
    inter_y_max = torch.min(y1_max, y2_max)

    inter_area = torch.clamp(inter_x_max - inter_x_min, min=0) * torch.clamp(inter_y_max - inter_y_min, min=0)
    union_area = (x1_max - x1_min) * (y1_max - y1_min) + (x2_max - x2_min) * (y2_max - y2_min) - inter_area

    return inter_area / (union_area + 1e-6)

# Validation Function
def validate_one_epoch(model, data_loader, device, num_classes):
    model.eval()
    focal_loss_fn = FocalLoss()
    total_loss = 0
    start_time = time.time()

    progress_bar = tqdm(data_loader, desc="Validating", leave=False)
    with torch.no_grad():
        for batch_idx, batch in enumerate(progress_bar):
            images = batch['pixel_values'].to(device)
            targets = batch['labels']

            target_boxes = [t['boxes'].to(device) for t in targets]
            target_labels = [t['class_labels'].to(device) for t in targets]

            pred_boxes, pred_logits = model(images)

            indices = hungarian_matcher(pred_boxes, pred_logits, target_boxes, target_labels, num_classes)

            loss_bbox = 0
            loss_cls = 0
            valid_batches = 0
            for i, (pred_idx, tgt_idx) in enumerate(indices):
                if len(tgt_idx) == 0:
                    continue
                matched_pred_boxes = pred_boxes[i, pred_idx]
                matched_target_boxes = target_boxes[i][tgt_idx]
                matched_pred_logits = pred_logits[i, pred_idx]
                matched_target_labels = F.one_hot(target_labels[i][tgt_idx].clamp(0, num_classes-1), num_classes=num_classes).float().to(device)

                loss_bbox += giou_loss(matched_pred_boxes, matched_target_boxes).mean()
                loss_cls += focal_loss_fn(matched_pred_logits, matched_target_labels)
                valid_batches += 1

            if valid_batches > 0:
                loss_bbox = loss_bbox / valid_batches
                loss_cls = loss_cls / valid_batches
                loss = loss_bbox + loss_cls
            else:
                loss = torch.tensor(0.0, device=device)

            total_loss += loss.item()
            progress_bar.set_postfix({"Loss": f"{loss.item():.4f}"})

    progress_bar.close()
    epoch_time = time.time() - start_time

    return total_loss / len(data_loader), epoch_time

# Evaluation Function for Testing
def evaluate_model(model, data_loader, device, num_classes, conf_thres=0.5, iou_thres=0.5, nms_iou_thres=0.45):
    model.eval()
    total_time = 0
    num_samples = 0
    all_preds = []
    all_gts = []
    img_id = 0

    TP = 0
    FP = 0
    FN = 0

    progress_bar = tqdm(data_loader, desc="Evaluating", leave=False)

    with torch.no_grad():
        for batch in progress_bar:
            start_time = time.time()
            images = batch['pixel_values'].to(device)
            targets = batch['labels']

            pred_boxes, pred_logits = model(images)
            pred_scores = pred_logits.softmax(-1)

            batch_size = images.shape[0]
            num_samples += batch_size

            for i in range(batch_size):
                scores = pred_scores[i]
                boxes = pred_boxes[i]
                target_boxes = targets[i]['boxes'].to(device)
                target_labels = targets[i]['class_labels'].to(device)

                # Apply confidence threshold and NMS
                max_scores, max_labels = scores.max(-1)
                valid_idx = max_scores > conf_thres
                valid_boxes = boxes[valid_idx]
                valid_scores = max_scores[valid_idx]
                valid_labels = max_labels[valid_idx]

                if len(valid_boxes) > 0:
                    keep_idx = nms(valid_boxes, valid_scores, nms_iou_thres)
                    valid_boxes = valid_boxes[keep_idx]
                    valid_scores = valid_scores[keep_idx]
                    valid_labels = valid_labels[keep_idx]

                # Prepare predictions for COCO format
                for box, score, label in zip(valid_boxes.cpu().numpy(), valid_scores.cpu().numpy(), valid_labels.cpu().numpy()):
                    x, y, w, h = box
                    all_preds.append({
                        'image_id': img_id,
                        'category_id': int(label) + 1,  # COCO expects 1-based indexing
                        'bbox': [float(x - w/2), float(y - h/2), float(w), float(h)],
                        'score': float(score)
                    })

                # Prepare ground truths for COCO format
                for box, label in zip(target_boxes.cpu().numpy(), target_labels.cpu().numpy()):
                    x, y, w, h = box
                    all_gts.append({
                        'image_id': img_id,
                        'category_id': int(label) + 1,  # COCO expects 1-based indexing
                        'bbox': [float(x - w/2), float(y - h/2), float(w), float(h)],
                        'id': len(all_gts) + 1,
                        'area': float(w * h),
                        'iscrowd': 0
                    })

                # Compute TP, FP, FN for Precision/Recall
                if len(valid_boxes) > 0 and len(target_boxes) > 0:
                    ious = compute_iou(valid_boxes.unsqueeze(1), target_boxes.unsqueeze(0))
                    max_ious, max_idx = ious.max(dim=1)
                    matched = max_ious > iou_thres

                    for j, match in enumerate(matched):
                        if match and valid_labels[j] == target_labels[max_idx[j]]:
                            TP += 1
                        else:
                            FP += 1

                    matched_gts = max_ious > iou_thres
                    FN += len(target_boxes) - matched_gts.sum()
                elif len(valid_boxes) == 0:
                    FN += len(target_boxes)
                elif len(target_boxes) == 0:
                    FP += len(valid_boxes)

                img_id += 1

            end_time = time.time()
            total_time += end_time - start_time

    # Compute Precision, Recall, F1-Score
    precision = TP / (TP + FP + 1e-6)
    recall = TP / (TP + FN + 1e-6)
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-6)

    # Compute mAP using pycocotools
    coco_gt = COCO()
    coco_gt.dataset = {
        'info': {},  # Add the 'info' key
        'images': [{'id': i} for i in range(img_id)],
        'annotations': all_gts,
        'categories': [{'id': i+1} for i in range(num_classes)]
    }
    coco_gt.createIndex()

    coco_dt = coco_gt.loadRes(all_preds)
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    map_50 = coco_eval.stats[1]  # mAP@0.5
    map_5095 = coco_eval.stats[0]  # mAP@0.5:0.95

    avg_time = total_time / num_samples if num_samples > 0 else 0

    progress_bar.close()

    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'mAP@0.5': map_50,
        'mAP@0.5:0.95': map_5095,
        'avg_inference_time': avg_time
    }

# Training Pipeline
def train_one_epoch(model, optimizer, data_loader, device, num_classes):
    model.train()
    focal_loss_fn = FocalLoss()
    total_loss = 0
    start_time = time.time()

    progress_bar = tqdm(data_loader, desc="Training", leave=False)
    for batch_idx, batch in enumerate(progress_bar):
        images = batch['pixel_values'].to(device)
        targets = batch['labels']

        target_boxes = [t['boxes'].to(device) for t in targets]
        target_labels = [t['class_labels'].to(device) for t in targets]

        optimizer.zero_grad()
        pred_boxes, pred_logits = model(images)

        indices = hungarian_matcher(pred_boxes, pred_logits, target_boxes, target_labels, num_classes)

        loss_bbox = 0
        loss_cls = 0
        valid_batches = 0
        for i, (pred_idx, tgt_idx) in enumerate(indices):
            if len(tgt_idx) == 0:
                continue
            matched_pred_boxes = pred_boxes[i, pred_idx]
            matched_target_boxes = target_boxes[i][tgt_idx]
            matched_pred_logits = pred_logits[i, pred_idx]
            matched_target_labels = F.one_hot(target_labels[i][tgt_idx].clamp(0, num_classes-1), num_classes=num_classes).float().to(device)

            loss_bbox += giou_loss(matched_pred_boxes, matched_target_boxes).mean()
            loss_cls += focal_loss_fn(matched_pred_logits, matched_target_labels)
            valid_batches += 1

        if valid_batches > 0:
            loss_bbox = loss_bbox / valid_batches
            loss_cls = loss_cls / valid_batches
            loss = loss_bbox + loss_cls
        else:
            loss = torch.tensor(0.0, requires_grad=True, device=device)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        progress_bar.set_postfix({"Loss": f"{loss.item():.4f}"})

    progress_bar.close()
    epoch_time = time.time() - start_time

    return total_loss / len(data_loader), epoch_time

# Plotting Function for Training and Validation Loss
def plot_losses(train_losses, valid_losses, num_epochs):
    plt.figure(figsize=(10, 5))
    epochs = range(1, num_epochs + 1)
    plt.plot(epochs, train_losses, label='Training Loss', marker='o')
    plt.plot(epochs, valid_losses, label='Validation Loss', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig('loss_plot.png')
    plt.show()