import torch
import torch.nn as nn
import torch.nn.functional as F

from models.cnn_model import CustomCNN


def get_model(model_name: str, num_classes: int, pretrained: bool = True, dropout_rate: float = 0.5):
    if model_name == 'customcnn':
        return CustomCNN(num_classes=num_classes)
    else:
        raise ValueError(f"Model {model_name} is not supported.")