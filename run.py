import argparse
import torch
import torch.nn as nn
import logging
import os

from utils.data_loader import create_data_loaders
from utils.train_evaluation import Trainer
from models.model_factory import get_model, FocalLoss, LabelSmoothingLoss
from sklearn.utils.class_weight import compute_class_weight
import numpy as np


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


CLASSES = ['Blight', 'Healthy_Leaf', 'Helopeltis', 'Red_Rust']

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = len(CLASSES)
    logger.info(f"Using device: {device}")
    logger.info(f"Number of classes: {num_classes}")

    # Load Data
    train_loader, val_loader, test_loader, class_weights = create_data_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        class_names=CLASSES
    )

    train_labels = train_loader.dataset.targets
    class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
    class_weights = torch.tensor(class_weights, dtype=torch.float32)
    logger.info(f"Class weights: {class_weights.numpy()}")

    # Get Model from Factory
    model = get_model(model_name=args.model_name, num_classes=num_classes)
    model = model.to(device)

    if args.loss == "focal":
        criterion = FocalLoss(alpha=1.0, gamma=2.0)
    elif args.loss == "labelsmoothing":
        criterion = LabelSmoothingLoss(num_classes=num_classes, smoothing=0.1)
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    # Initialize Trainer
    trainer = Trainer(
        model=model, 
        device=device, 
        class_names=CLASSES, 
        model_name=args.model_name
    )

    # Run Training
    logger.info(f"Starting training with {args.model_name} and {args.loss} loss...")
    trainer.train(
        train_loader=train_loader, 
        val_loader=val_loader, 
        epochs=args.epochs,
        criterion=criterion,
        lr=args.lr
    )


    test_metrics, test_loss = trainer.evaluate(test_loader) 
    logger.info(f"Final Results - Accuracy: {test_metrics['accuracy']:.4f}")
    logger.info(f"Final Results - Loss: {test_loss:.4f}")
    
def parse_args():
    parser = argparse.ArgumentParser(description="Tea Leaf Disease Standalone Training")
    
    # Data & Architecture
    parser.add_argument("--data-dir", type=str, required=True, help="Path to dataset")
    parser.add_argument("--model-name", type=str, default="customcnn", choices=["customcnn", "vitb16", "efficientnetb3", "mobilenetv3"], help="Model from factory")
    parser.add_argument("--batch-size", type=int, default=32)
    
    # Training Hyperparameters
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--loss", type=str, default="crossentropy", choices=["crossentropy", "focal", "labelsmoothing"])
    
    # Early Stopping & Saving
    parser.add_argument("--patience", type=int, default=10, help="Patience for early stopping")
    parser.add_argument("--save-dir", type=str, default="checkpoints")
    
    return parser.parse_args()

if __name__ == "__main__":
    main()