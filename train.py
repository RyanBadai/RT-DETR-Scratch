import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
import time
from torchvision.ops import nms
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

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