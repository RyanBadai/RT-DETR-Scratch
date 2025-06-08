import torch
import torch.nn as nn

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