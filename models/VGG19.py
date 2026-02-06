import torch.nn as nn
from torchvision import models

class VGG19Model(nn.Module):
    def __init__(self, num_classes: int = 4, dropout_rate: float = 0.5):
        super(VGG19Model, self).__init__()
        self.backbone = models.vgg19(weights='DEFAULT')
        
        # VGG19 classifier starts at index 0
        in_features = self.backbone.classifier[0].in_features
        
        # Replace the entire classifier block for better regularization
        self.backbone.classifier = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate * 0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)