import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm
import time
from torchvision.ops import nms, box_convert
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import supervision as sv
# MODIFIKASI: Import fungsi untuk mengunduh dari URL
from torch.hub import load_state_dict_from_url

# ===================================================================
# DEFINISI MODEL (Tidak ada perubahan)
# ===================================================================
class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)
    def forward(self, x):
        return x * self.relu(x + 3) / 6

class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        mip = max(8, inp // reduction)
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x; n,c,h,w = x.size()
        x_h = self.pool_h(x); x_w = self.pool_w(x).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y); y = self.bn1(y); y = self.act(y) 
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        a_h = self.conv_h(x_h).sigmoid(); a_w = self.conv_w(x_w).sigmoid()
        out = identity * a_w * a_h
        return out

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x))); out = self.bn2(self.conv2(out))
        out += self.shortcut(x); out = self.relu(out)
        return out

class CoordAttBasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(CoordAttBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.coord_att = CoordAtt(out_channels, out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x))); out = self.bn2(self.conv2(out))
        out = self.coord_att(out); out += self.shortcut(x); out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(BasicBlock, 64, 128, 2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, 256, 2, stride=2)
        self.layer3 = self._make_layer(CoordAttBasicBlock, 256, 512, 2, stride=2)
        self.layer4 = self._make_layer(CoordAttBasicBlock, 512, 1024, 2, stride=2)
        self.adjust_s4 = nn.Conv2d(256, 128, kernel_size=1)
        self.adjust_s5 = nn.Conv2d(512, 256, kernel_size=1)
        self.adjust_s6 = nn.Conv2d(1024, 512, kernel_size=1)
    def _make_layer(self, block_type, in_channels, out_channels, blocks, stride):
        layers = []; layers.append(block_type(in_channels, out_channels, stride))
        for _ in range(1, blocks): layers.append(block_type(out_channels, out_channels, 1))
        return nn.Sequential(*layers)
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x))); x = self.maxpool(x)
        s3 = self.layer1(x); s4 = self.layer2(s3)
        s5 = self.layer3(s4); s6 = self.layer4(s5)
        feat1 = self.adjust_s4(s4); feat2 = self.adjust_s5(s5)
        feat3 = self.adjust_s6(s6)
        return [feat1, feat2, feat3]

class AIFI(nn.Module):
    def __init__(self, dim):
        super(AIFI, self).__init__(); self.attention = nn.MultiheadAttention(embed_dim=dim, num_heads=8, batch_first=True)
        self.norm = nn.LayerNorm(dim); self.ffn = nn.Sequential(nn.Linear(dim, dim * 4), nn.ReLU(), nn.Linear(dim * 4, dim)); self.norm2 = nn.LayerNorm(dim)
    def forward(self, x):
        B, C, H, W = x.shape; x_reshaped = x.view(B, C, H * W).permute(0, 2, 1)
        attn_output, _ = self.attention(x_reshaped, x_reshaped, x_reshaped)
        x_reshaped = self.norm(x_reshaped + attn_output); ffn_output = self.ffn(x_reshaped)
        x_reshaped = self.norm2(x_reshaped + ffn_output); x = x_reshaped.permute(0, 2, 1).view(B, C, H, W); return x

class RTDETRHybridEncoder(nn.Module):
    def __init__(self, channels=[128, 256, 512]):
        super(RTDETRHybridEncoder, self).__init__(); self.aifi = AIFI(dim=channels[2])
        self.fusion_s5 = nn.Conv2d(channels[2] + channels[1], channels[1], kernel_size=3, padding=1); self.norm_s5 = nn.BatchNorm2d(channels[1])
        self.fusion_s4 = nn.Conv2d(channels[1] + channels[0], channels[0], kernel_size=3, padding=1); self.norm_s4 = nn.BatchNorm2d(channels[0]); self.relu = nn.ReLU(inplace=True)
    def forward(self, features):
        s4, s5, s6 = features; s6_enhanced = self.aifi(s6)
        s6_upsampled = F.interpolate(s6_enhanced, size=s5.shape[2:], mode='bilinear', align_corners=False)
        fused_s5 = torch.cat([s6_upsampled, s5], dim=1); fused_s5 = self.relu(self.norm_s5(self.fusion_s5(fused_s5)))
        s5_upsampled = F.interpolate(fused_s5, size=s4.shape[2:], mode='bilinear', align_corners=False)
        fused_s4 = torch.cat([s5_upsampled, s4], dim=1); fused_s4 = self.relu(self.norm_s4(self.fusion_s4(fused_s4)))
        return [fused_s4, fused_s5, s6_enhanced]

class IoUQuerySelection(nn.Module):
    def __init__(self, dim, num_queries=300):
        super(IoUQuerySelection, self).__init__(); self.query_embed = nn.Embedding(num_queries, dim); self.fc = nn.Linear(dim, dim)
    def forward(self, x):
        B, C, _, _ = x.shape; queries = self.query_embed.weight.unsqueeze(0).repeat(B, 1, 1); queries = self.fc(queries); return queries

class RTDETRDecoder(nn.Module):
    def __init__(self, dim, num_queries=300, num_classes=3):
        super(RTDETRDecoder, self).__init__()
        self.layers = nn.ModuleList([nn.TransformerDecoderLayer(d_model=dim, nhead=8, batch_first=True) for _ in range(6)])
        self.query_selection = IoUQuerySelection(dim, num_queries); self.fc_bbox = nn.Linear(dim, 4); self.fc_cls = nn.Linear(dim, num_classes)
    def forward(self, memory):
        B, C, H, W = memory.shape; queries = self.query_selection(memory); memory = memory.view(B, C, H * W).permute(0, 2, 1)
        for layer in self.layers: queries = layer(queries, memory)
        bboxes = self.fc_bbox(queries).sigmoid(); cls_scores = self.fc_cls(queries)
        return bboxes, cls_scores

class RTDETR(nn.Module):
    def __init__(self, num_classes=3):
        super(RTDETR, self).__init__(); self.backbone = ResNet(); self.encoder = RTDETRHybridEncoder(channels=[128, 256, 512]) 
        self.decoder = RTDETRDecoder(dim=512, num_classes=num_classes)
    def forward(self, x):
        features = self.backbone(x); memory_pyramid = self.encoder(features); memory = memory_pyramid[-1]; bboxes, cls_scores = self.decoder(memory)
        return bboxes, cls_scores

# ===================================================================
# MODIFIKASI: FUNGSI BARU UNTUK MEMUAT PRE-TRAINED WEIGHTS
# ===================================================================
def load_model_with_pretrained_weights(num_classes, weights_url='https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r50vd_6x_coco_from_paddle.pth', device='cpu'):
    """
    Membuat model RTDETR, mengunduh bobot pre-trained dari URL,
    dan memuatnya ke dalam model, dengan mengabaikan kepala klasifikasi.
    """
    print(f"Membuat model kustom untuk {num_classes} kelas...")
    # Buat model dengan jumlah kelas yang benar di decoder.
    # Model asli Anda tidak memiliki +1 untuk 'no object', jadi kita ikuti itu.
    model = RTDETR(num_classes=num_classes)
    
    try:
        print(f"Mengunduh bobot pre-trained dari: {weights_url}")
        pretrained_state_dict = load_state_dict_from_url(weights_url, map_location=device, progress=True)
        print("Unduhan bobot berhasil.")
    except Exception as e:
        print(f"Gagal mengunduh atau memuat bobot dari URL: {e}")
        print("Mengembalikan model tanpa bobot pre-trained.")
        return model

    model_state_dict = model.state_dict()
    weights_to_load = {}
    
    for name, param in pretrained_state_dict.items():
        # Menghapus awalan 'model.' yang sering ada pada bobot pre-trained
        clean_name = name.replace("model.", "")
        
        if clean_name in model_state_dict:
            # Muat hanya jika nama dan ukuran cocok
            if param.shape == model_state_dict[clean_name].shape:
                weights_to_load[clean_name] = param
            else:
                print(f"  -> Mengabaikan (ukuran tidak cocok): {clean_name}")
        # else:
            # print(f"  -> Mengabaikan (layer tidak ditemukan): {clean_name}")

    # Muat bobot yang sudah difilter. `strict=False` penting karena
    # kita sengaja mengabaikan kepala klasifikasi.
    model.load_state_dict(weights_to_load, strict=False)
    
    print("\n✅ Berhasil memuat bobot pre-trained.")
    print("   Kepala klasifikasi ('decoder.fc_cls') akan dilatih dari awal pada dataset Anda.")
    
    return model

# ===================================================================
# LOSS, MATCHER, DAN PIPELINE (Tidak ada perubahan)
# ===================================================================
class WShapeIoULoss(nn.Module):
    def __init__(self, scale=4.0, d=0.0, u=0.95, r=0.5):
        super(WShapeIoULoss, self).__init__(); self.scale = scale; self.d = d; self.u = u; self.r = r; self.eps = 1e-7
    def forward(self, pred, target):
        b1_x1, b1_y1 = pred[..., 0] - pred[..., 2] / 2, pred[..., 1] - pred[..., 3] / 2
        b1_x2, b1_y2 = pred[..., 0] + pred[..., 2] / 2, pred[..., 1] + pred[..., 3] / 2
        b2_x1, b2_y1 = target[..., 0] - target[..., 2] / 2, target[..., 1] - target[..., 3] / 2
        b2_x2, b2_y2 = target[..., 0] + target[..., 2] / 2, target[..., 1] + target[..., 3] / 2
        inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)
        w1, h1 = (b1_x2 - b1_x1), (b1_y2 - b1_y1); w2, h2 = (b2_x2 - b2_x1), (b2_y2 - b2_y1)
        union = w1 * h1 + w2 * h2 - inter
        iou = inter / union.clamp(min=self.eps)
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1); ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)
        c2 = (cw**2 + ch**2).clamp(min=self.eps)
        rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2)**2 + (b2_y1 + b2_y2 - b1_y1 - b1_y2)**2) / 4
        ww = (2 * (w2**self.scale)) / (w2**self.scale + h2**self.scale).clamp(min=self.eps)
        hh = (2 * (h2**self.scale)) / (w2**self.scale + h2**self.scale).clamp(min=self.eps)
        distance_shape = (hh * rho2 + ww * rho2) / c2
        omega_w = hh * torch.abs(w1 - w2) / torch.max(w1, w2).clamp(min=self.eps)
        omega_h = ww * torch.abs(h1 - h2) / torch.max(h1, h2).clamp(min=self.eps)
        shape_value_loss = (1 - torch.exp(-omega_w))**4 + (1 - torch.exp(-omega_h))**4
        loss_shapeiou = 1 - iou + distance_shape + 0.5 * shape_value_loss
        iou_focaler = (1 - (iou - self.d) / (self.u - self.d)).clamp(0, 1)
        iou_score = iou.detach()
        beta = iou_focaler * (iou_score / iou_score.mean().clamp(min=self.eps))**self.r
        return (beta * loss_shapeiou).mean()

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__(); self.alpha = alpha; self.gamma = gamma
    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss); F_loss = self.alpha * (1 - pt)**self.gamma * BCE_loss
        return F_loss.mean()

def hungarian_matcher(pred_boxes, pred_logits, target_boxes_list, target_labels_list, num_classes):
    B = pred_boxes.shape[0]; indices = []; wshape_loss_fn = WShapeIoULoss()
    for i in range(B):
        tgt_boxes, tgt_labels = target_boxes_list[i], target_labels_list[i]
        valid_gt_mask = (tgt_boxes[:, 2] > 0) & (tgt_boxes[:, 3] > 0)
        tgt_boxes, tgt_labels = tgt_boxes[valid_gt_mask], tgt_labels[valid_gt_mask]
        if len(tgt_boxes) == 0:
            indices.append((torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long))); continue
        out_prob = pred_logits[i].softmax(-1)
        valid_labels_mask = tgt_labels < num_classes
        if not valid_labels_mask.any():
            indices.append((torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long))); continue
        valid_labels = tgt_labels[valid_labels_mask]
        cost_class = -out_prob[:, valid_labels]
        cost_bbox = torch.cdist(pred_boxes[i], tgt_boxes[valid_labels_mask], p=1)
        with torch.no_grad():
            cost_wshape = wshape_loss_fn(pred_boxes[i].unsqueeze(1), tgt_boxes[valid_labels_mask].unsqueeze(0)).squeeze(0)
        C = cost_bbox + 0.5 * cost_wshape + cost_class
        C = torch.nan_to_num(C, nan=1e5, posinf=1e5, neginf=1e5)
        pred_idx, tgt_idx = linear_sum_assignment(C.cpu().detach().numpy())
        indices.append((torch.as_tensor(pred_idx, dtype=torch.long), torch.as_tensor(tgt_idx, dtype=torch.long)))
    return indices

def train_one_epoch(model, optimizer, data_loader, device, num_classes):
    model.train(); focal_loss_fn = FocalLoss(); bbox_loss_fn = WShapeIoULoss()
    total_loss = 0; start_time = time.time()
    progress_bar = tqdm(data_loader, desc="Training", leave=False)
    for batch in progress_bar:
        images = batch['pixel_values'].to(device)
        targets = batch['labels']
        target_boxes = [t['boxes'].to(device) for t in targets]
        target_labels = [t['class_labels'].to(device) for t in targets]
        optimizer.zero_grad()
        pred_boxes, pred_logits = model(images)
        indices = hungarian_matcher(pred_boxes, pred_logits, target_boxes, target_labels, num_classes)
        loss_bbox = 0; loss_cls = 0; valid_batches = 0
        for i, (pred_idx, tgt_idx) in enumerate(indices):
            if len(tgt_idx) == 0: continue
            valid_gt_mask = (target_boxes[i][:, 2] > 0) & (target_boxes[i][:, 3] > 0)
            if not valid_gt_mask.any(): continue
            matched_pred_boxes = pred_boxes[i, pred_idx]
            matched_target_boxes = target_boxes[i][valid_gt_mask][tgt_idx]
            matched_pred_logits = pred_logits[i, pred_idx]
            target_one_hot = F.one_hot(target_labels[i][valid_gt_mask][tgt_idx].long(), num_classes=num_classes).float()
            loss_bbox += bbox_loss_fn(matched_pred_boxes, matched_target_boxes)
            loss_cls += focal_loss_fn(matched_pred_logits, target_one_hot)
            valid_batches += 1
        
        if valid_batches > 0:
            loss = 5 * (loss_bbox / valid_batches) + 2 * (loss_cls / valid_batches)
        else: loss = torch.tensor(0.0, device=device, requires_grad=True)

        if torch.isfinite(loss):
            loss.backward(); optimizer.step(); total_loss += loss.item()
            progress_bar.set_postfix({"Loss": f"{loss.item():.4f}"})
        else: print("Peringatan: Loss tidak valid terdeteksi, update dilewati.")
    
    progress_bar.close(); return total_loss / len(data_loader), time.time() - start_time

def validate_one_epoch(model, data_loader, device, num_classes):
    model.eval(); focal_loss_fn = FocalLoss(); bbox_loss_fn = WShapeIoULoss()
    total_loss = 0; start_time = time.time()
    progress_bar = tqdm(data_loader, desc="Validating", leave=False)
    with torch.no_grad():
        for batch in progress_bar:
            images = batch['pixel_values'].to(device)
            targets = batch['labels']
            target_boxes = [t['boxes'].to(device) for t in targets]
            target_labels = [t['class_labels'].to(device) for t in targets]
            pred_boxes, pred_logits = model(images)
            indices = hungarian_matcher(pred_boxes, pred_logits, target_boxes, target_labels, num_classes)
            loss_bbox = 0; loss_cls = 0; valid_batches = 0
            for i, (pred_idx, tgt_idx) in enumerate(indices):
                if len(tgt_idx) == 0: continue
                valid_gt_mask = (target_boxes[i][:, 2] > 0) & (target_boxes[i][:, 3] > 0)
                if not valid_gt_mask.any(): continue
                matched_pred_boxes = pred_boxes[i, pred_idx]
                matched_target_boxes = target_boxes[i][valid_gt_mask][tgt_idx]
                matched_pred_logits = pred_logits[i, pred_idx]
                target_one_hot = F.one_hot(target_labels[i][valid_gt_mask][tgt_idx].long(), num_classes=num_classes).float()
                loss_bbox += bbox_loss_fn(matched_pred_boxes, matched_target_boxes)
                loss_cls += focal_loss_fn(matched_pred_logits, target_one_hot)
                valid_batches += 1
            if valid_batches > 0:
                loss = 5 * (loss_bbox / valid_batches) + 2 * (loss_cls / valid_batches)
                total_loss += loss.item()
                progress_bar.set_postfix({"Val Loss": f"{loss.item():.4f}"})

    progress_bar.close(); return total_loss / len(data_loader), time.time() - start_time

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