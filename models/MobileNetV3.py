import torch
import torch.nn as nn
from torchvision import models

class MobileNetV3(nn.Module):
    def __init__(self, num_classes=4, pretrained=True):
        super(MobileNetV3, self).__init__()
        # Load pre-trained MobileNetV3-Large for high-quality RGB features
        self.backbone = models.mobilenet_v3_large(pretrained=pretrained)
        
        # Access the input features for the original classifier
        num_features = self.backbone.classifier[0].in_features
        self.backbone.classifier = nn.Identity() # Remove the default head

        # Specialized MLP Head for Tea Diseases
        self.classifier = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.Hardswish(inplace=True), # MobileNetV3 native activation
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.Hardswish(inplace=True),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)

