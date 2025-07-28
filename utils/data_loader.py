import os
import torch
import numpy as np
from PIL import Image
import cv2
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Tuple, List, Optional, Dict
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TeaLeafDataset(Dataset):
    def __init__(self, data_dir: str, transform: None, subset: str = 'train'):
        self.data_dir = data_dir
        self.transform = transform
        self.subset = subset

        self.class_names = sorted([
            data for data in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, data))
        ])

        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.class_names)}
        self.idx_to_class = {idx: cls_name for cls_name, idx in self.class_to_idx.items()}

        logger.info("Class to index mapping")
        for cls_name, idx in self.class_to_idx.items():
            logger.info(f"{cls_name}: {idx}")
        
        # Load images and labels
        self.samples = self._load_samples()

        #calculate class weights
        self.class_weights = self._calculate_class_weights()

        logger.info(f"Total samples in {subset} set: {len(self.samples)}")
        self._log_class_distribution()

    def _load_samples(self) -> List[Tuple[str, int]]:
        """Load all image paths and their corresponding labels"""
        samples = []

        for class_names in os.listdir(self.data_dir):
            class_path = os.path.join(self.data_dir, class_names)
            if os.path.isdir(class_path):
                for img_name in os.listdir(class_path):
                    img_path = os.path.join(class_path, img_name)
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        samples.append((img_path, self.class_to_idx[class_names]))
        return samples
    
    