import torch
import torch.nn as nn
from backbone import ResNet
from encoder import EfficientHybridEncoder
from decoder import RTDETRDecoder

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