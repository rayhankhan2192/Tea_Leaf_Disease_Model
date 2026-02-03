import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import json
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from tqdm import tqdm

class EarlyStopping:
    """Stops training when validation loss stops improving"""
    def __init__(self, patience=7, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience: 
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        return self.early_stop

class Metrics:
    def __init__(self, class_names):
        self.class_names = class_names

    def calculate_metrics(self, y_true, y_pred):
        """Calculates standard and per-class tea disease metrics"""
        acc = accuracy_score(y_true, y_pred)
        p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
        
        metrics = {'accuracy': acc, 'f1_macro': f1}
        
        cm = confusion_matrix(y_true, y_pred, labels=range(len(self.class_names)))
        for i, name in enumerate(self.class_names):
            tp = cm[i, i]
            fn = cm[i, :].sum() - tp
            fp = cm[:, i].sum() - tp
            tn = cm.sum() - (tp + fn + fp)
            metrics[f"{name.lower()}_sensitivity"] = tp / (tp + fn) if (tp + fn) > 0 else 0
            metrics[f"{name.lower()}_specificity"] = tn / (tn + fp) if (tn + fp) > 0 else 0
            
        return metrics

class Trainer:
    def __init__(self, model, device, class_names, save_dir='checkpoints'):
        self.model = model.to(device)
        self.device = device
        self.class_names = class_names
        self.metrics_calc = Metrics(class_names)
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def train(self, train_loader, val_loader, epochs=50):
        optimizer = optim.AdamW(self.model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        early_stop = EarlyStopping(patience=10)
        
        best_acc = 0.0

        for epoch in range(epochs):
            self.model.train()
            for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                loss = criterion(self.model(imgs), labels)
                loss.backward()
                optimizer.step()

            val_results = self.evaluate(val_loader)
            print(f"Epoch {epoch+1} | Val Acc: {val_results['accuracy']:.4f}")

            if val_results['accuracy'] > best_acc:
                best_acc = val_results['accuracy']
                torch.save(self.model.state_dict(), os.path.join(self.save_dir, 'best_model.pth'))

            if early_stop(val_results['accuracy']): 
                print("Early stopping triggered.")
                break

    def evaluate(self, loader):
        self.model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for imgs, labels in loader:
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                outputs = self.model(imgs)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        return self.metrics_calc.calculate_metrics(np.array(all_labels), np.array(all_preds))