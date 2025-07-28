import os
import cv2
import torch
import numpy as np
import logging
from PIL import Image
from torch.utils.data import Dataset
from typing import Tuple, List, Optional, Dict
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Setup logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TeaLeafDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        transform: Optional[callable] = None,
        subset: str = 'train',
        image_size: Tuple[int, int] = (224, 224)
    ):
        self.data_dir = data_dir
        self.transform = transform
        self.subset = subset
        self.image_size = image_size

        # Get class names
        self.class_names = sorted([
            entry for entry in os.listdir(data_dir)
            if os.path.isdir(os.path.join(data_dir, entry))
        ])

        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.class_names)}
        self.idx_to_class = {idx: cls_name for cls_name, idx in self.class_to_idx.items()}

        logger.info("Class to index mapping:")
        for cls_name, idx in self.class_to_idx.items():
            logger.info(f"  {cls_name}: {idx}")

        self.samples = self._load_samples()
        self.class_weights = self._calculate_class_weights()

        logger.info(f"Total samples in '{subset}' set: {len(self.samples)}")
        self._log_class_distribution()
        logger.info(f"Class Weights (Tensor): {self.class_weights}")

    def _load_samples(self) -> List[Tuple[str, int]]:
        """Load all image paths and corresponding labels"""
        samples = []
        for class_name in self.class_names:
            class_path = os.path.join(self.data_dir, class_name)
            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    samples.append((img_path, self.class_to_idx[class_name]))
        return samples

    def _calculate_class_weights(self) -> torch.Tensor:
        """Calculate balanced class weights"""
        class_counts = np.zeros(len(self.class_names), dtype=np.int64)
        for _, label in self.samples:
            class_counts[label] += 1

        total_samples = len(self.samples)
        class_weights = []
        for count in class_counts:
            if count == 0:
                class_weights.append(0.0)
                logger.warning("Found a class with zero samples, assigning weight 0.0")
            else:
                weight = total_samples / (len(self.class_names) * count)
                class_weights.append(weight)
        return torch.tensor(class_weights, dtype=torch.float32)

    def _log_class_distribution(self):
        """Log the distribution of samples per class"""
        class_counts = np.zeros(len(self.class_names), dtype=np.int64)
        for _, label in self.samples:
            class_counts[label] += 1

        logger.info("Class distribution:")
        for cls_name, count in zip(self.class_names, class_counts):
            logger.info(f"  {cls_name}: {count}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        image = self._load_and_preprocess_image(img_path)

        if self.transform:
            image = self.transform(image=image)['image']

        return image, label

    def _load_and_preprocess_image(self, img_path: str) -> np.ndarray:
        """Safely load an image and resize/normalize it"""
        try:
            image = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if image is None:
                raise FileNotFoundError(f"Unable to read image at: {img_path}")

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, self.image_size)
            image = image.astype(np.float32) / 255.0
            return image

        except Exception as e:
            logger.error(f"Error loading image {img_path}: {e}")
            return np.zeros((*self.image_size, 3), dtype=np.float32)



def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True, help="Path to tea leaf dataset")
    args = parser.parse_args()

    # ✅ Define transform
    transform = A.Compose([
        A.Resize(256, 256),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    dataset = TeaLeafDataset(data_dir=args.data_dir, transform=transform, subset='train')

if __name__ == "__main__":
    main()