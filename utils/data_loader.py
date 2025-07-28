import os
import cv2
import torch
import numpy as np
import logging
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from typing import Tuple, List, Optional, Dict
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split


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
        logger.info(f"Class Weights (Tensor): {self.class_weights}\n")

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

def get_tea_leaf_transforms(image_size=(224, 224), mode='train'):
    if mode == 'train':
        return A.Compose([
            A.Rotate(limit=15, p=0.7),
            A.RandomScale(scale_limit=0.1, p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.Resize(*image_size),
            A.Normalize(mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(*image_size),
            A.Normalize(mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
def create_data_loaders(data_dir: str, 
                       batch_size: int = 32, 
                       train_split: float = 0.8,
                       val_split: float = 0.1,
                       test_split: float = 0.1,
                       image_size: Tuple[int, int] = (224, 224),
                       num_workers: int = 1,
                       pin_memory: bool = True,
                       seed: int = 42) -> Tuple[DataLoader, DataLoader, DataLoader, torch.Tensor]:
    assert abs(train_split + val_split + test_split - 1.0) < 1e-6, "Splits must sum to 1.0"

    train_transform = get_tea_leaf_transforms(image_size, mode='train')
    val_transform = get_tea_leaf_transforms(image_size, mode='val')
    test_transform = get_tea_leaf_transforms(image_size, mode='test')

    full_dataset = TeaLeafDataset(data_dir, transform=None, subset='full', image_size=image_size)
    total_size = len(full_dataset)

    train_size = int(train_split * total_size)
    val_size = int(val_split * total_size)
    test_size = total_size - train_size - val_size

    labels = [sample[1] for sample in full_dataset.samples]
    train_indices, temp_indices = train_test_split(
        range(total_size),
        train_size=train_size,
        stratify=labels,
        random_state=seed
    )

    temp_labels = [labels[i] for i in temp_indices]
    val_indices, test_indices = train_test_split(
        temp_indices,
        train_size=val_size,
        stratify=temp_labels,
        random_state=seed
    )

    train_samples = [full_dataset.samples[i] for i in train_indices]
    val_samples = [full_dataset.samples[i] for i in val_indices]
    test_samples = [full_dataset.samples[i] for i in test_indices]

    train_dataset = TeaLeafDataset(data_dir, transform=train_transform, subset='train', image_size=image_size)
    train_dataset.samples = train_samples

    val_dataset = TeaLeafDataset(data_dir, transform=val_transform, subset='val', image_size=image_size)
    val_dataset.samples = val_samples

    test_dataset = TeaLeafDataset(data_dir, transform=test_transform, subset='test', image_size=image_size)
    test_dataset.samples = test_samples

    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                              num_workers=num_workers, pin_memory=pin_memory, drop_last=True, generator=generator)

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                            num_workers=num_workers, pin_memory=pin_memory)

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                             num_workers=num_workers, pin_memory=pin_memory)

    logger.info(f"Data Split")
    logger.info(f"  - Train: {len(train_dataset)}")
    logger.info(f"  - Val:   {len(val_dataset)}")
    logger.info(f"  - Test:  {len(test_dataset)}")

    return train_loader, val_loader, test_loader, train_dataset.class_weights


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True, help="Path to tea leaf dataset")
    args = parser.parse_args()

    train_loader, val_loader, test_loader, class_weights = create_data_loaders(
        data_dir=args.data_dir,
        batch_size=32,
        train_split=0.8,
        val_split=0.1,
        test_split=0.1,
        image_size=(224, 224),
        num_workers=1,
        pin_memory=True
    )

if __name__ == "__main__":
    main()