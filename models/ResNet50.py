import torch.nn as nn
from torchvision import models

class ResNet50Model(nn.Module):
    def __init__(self, num_classes: int = 4, dropout_rate: float = 0.4):
        super(ResNet50Model, self).__init__()
        self.backbone = models.resnet50(weights='DEFAULT')
        in_features = self.backbone.fc.in_features
        
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)