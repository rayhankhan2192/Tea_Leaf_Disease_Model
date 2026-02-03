import argparse
import torch
import torch.nn as nn
import logging
import os

from utils.data_loader import create_data_loaders
from utils.train_evaluation import Trainer
from models.model_factory import get_model, FocalLoss, LabelSmoothingLoss

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


MY_CLASSES = ['Blight', 'Healthy_Leaf', 'Helopeltis', 'Red_Rust']

def parse_args():
    parser = argparse.ArgumentParser(description="Tea Leaf Disease Standalone Training")
    
    # Data & Architecture
    parser.add_argument("--data-dir", type=str, required=True, help="Path to dataset")
    parser.add_argument("--model-name", type=str, default="customcnn", choices=["customcnn"], help="Model from factory")
    parser.add_argument("--batch-size", type=int, default=32)
    
    # Training Hyperparameters
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--loss", type=str, default="crossentropy", choices=["crossentropy", "focal", "labelsmoothing"])
    
    # Early Stopping & Saving
    parser.add_argument("--patience", type=int, default=10, help="Patience for early stopping")
    parser.add_argument("--save-dir", type=str, default="checkpoints")
    
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = len(MY_CLASSES)

    # Load Data
    train_loader, val_loader, test_loader = create_data_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        class_names=MY_CLASSES
    )

    # Get Model from Factory
    model = get_model(model_name=args.model_name, num_classes=num_classes)
    model = model.to(device)

    # Select Loss Function from Factory
    if args.loss == "focal":
        criterion = FocalLoss(alpha=1.0, gamma=2.0)
    elif args.loss == "labelsmoothing":
        criterion = LabelSmoothingLoss(num_classes=num_classes, smoothing=0.1)
    else:
        criterion = nn.CrossEntropyLoss()

    # Initialize Trainer
    trainer = Trainer(
        model=model, 
        device=device, 
        class_names=MY_CLASSES, 
        save_dir=args.save_dir
    )

    # Run Training
    logger.info(f"Starting training with {args.model_name} and {args.loss} loss...")
    trainer.train(
        train_loader=train_loader, 
        val_loader=val_loader, 
        epochs=args.epochs,
        criterion=criterion 
    )

    test_metrics = trainer.evaluate(test_loader)
    logger.info(f"Final Results - Accuracy: {test_metrics['accuracy']:.4f}")

if __name__ == "__main__":
    main()