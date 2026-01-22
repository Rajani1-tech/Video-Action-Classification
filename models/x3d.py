# models/x3d.py
import torch.nn as nn
import torch
from pytorchvideo.models.hub import x3d_s

class X3DPlayClassifier(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.model = x3d_s(pretrained=True)
        self.model.head = nn.Sequential(
            nn.AdaptiveAvgPool3d((1,1,1)),
            nn.Flatten(),
            nn.Linear(2048, num_classes)
        )

    def forward(self, x):
        return self.model(x)
