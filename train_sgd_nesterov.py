"""
CNN Training with SGD + Nesterov Momentum
Optimized for MAXIMUM validation accuracy

This script uses SGD with Nesterov momentum which often achieves
better generalization than Adam/AdamW.

Expected improvement over Adam baseline: +2-3%
Target: 78-79% validation accuracy from your 76.61%
"""

import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import random

# Set random seeds
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# Import dataset and model from existing code
from train_model import TinyImageNetDataset, CustomCNN, EarlyStopping


class LabelSmoothingLoss(nn.Module):
    """Label smoothing loss"""
    def __init__(self, classes=15, smoothing=0.1):
        super().__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.classes = classes

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=-1))


def mixup_data(x, y, alpha=0.2, device='cuda'):
    """Mixup augmentation"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def train_epoch_with_mixup(model, dataloader, criterion, optimizer, device, use_mixup=True):
    """Train for one epoch with mixup"""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    for images, labels in tqdm(dataloader, desc='Training', leave=False):
        images, labels = images.to(device), labels.to(device)

        if use_mixup and random.random() > 0.5:
            images, targets_a, targets_b, lam = mixup_data(images, labels, alpha=0.2, device=device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
        else:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)

        loss.backward()

        # Gradient clipping for stability with SGD
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    return epoch_loss, epoch_acc


def validate_epoch(model, dataloader, criterion, device):
    """Validate for one epoch"""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc='Validation', leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    return epoch_loss, epoch_acc


def plot_training_history(train_losses, train_accs, val_losses, val_accs, save_path='training_history_sgd.png'):
    """Plot training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(train_losses, label='Train Loss', marker='o', alpha=0.7)
    ax1.plot(val_losses, label='Val Loss', marker='s', alpha=0.7)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss (SGD+Nesterov)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(train_accs, label='Train Acc', marker='o', alpha=0.7)
    ax2.plot(val_accs, label='Val Acc', marker='s', alpha=0.7)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracy (SGD+Nesterov)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f'Training history plot saved to {save_path}')
    plt.close()


def main():
    # ===== SGD OPTIMIZED HYPERPARAMETERS =====
    BATCH_SIZE = 64
    NUM_EPOCHS = 150

    # SGD hyperparameters (different from Adam!)
    LEARNING_RATE = 0.01       # Higher than Adam (Adam uses 0.001)
    MOMENTUM = 0.9             # Standard momentum
    WEIGHT_DECAY = 5e-4        # Stronger than Adam (Adam uses 1e-4)
    NESTEROV = True            # KEY: Use Nesterov momentum

    DROPOUT_RATE = 0.4
    PATIENCE = 20
    NUM_CLASSES = 15
    USE_MIXUP = True
    LABEL_SMOOTHING = 0.1

    TRAIN_PICKLE = 'train-70_.pkl'
    VAL_PICKLE = 'validation-10_.pkl'
    MODEL_SAVE_PATH = 'model_sgd_nesterov.pth'

    # Check data files
    if not os.path.exists(TRAIN_PICKLE):
        print(f"Error: {TRAIN_PICKLE} not found!")
        return
    if not os.path.exists(VAL_PICKLE):
        print(f"Error: {VAL_PICKLE} not found!")
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ENHANCED Data augmentation
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(20),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.15, 0.15),
            scale=(0.9, 1.1),
            shear=10
        ),
        transforms.ColorJitter(
            brightness=0.3,
            contrast=0.3,
            saturation=0.3,
            hue=0.15
        ),
        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=3)
        ], p=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load datasets
    print("\n" + "="*60)
    print("Loading Datasets")
    print("="*60)
    train_dataset = TinyImageNetDataset(TRAIN_PICKLE, transform=train_transform)
    val_dataset = TinyImageNetDataset(VAL_PICKLE, transform=val_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False
    )

    print(f"\nTrain batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")

    # Create model
    print("\n" + "="*60)
    print("Creating Model")
    print("="*60)
    model = CustomCNN(num_classes=NUM_CLASSES, dropout_rate=DROPOUT_RATE)
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # Label smoothing loss
    criterion = LabelSmoothingLoss(classes=NUM_CLASSES, smoothing=LABEL_SMOOTHING)

    # ===== SGD WITH NESTEROV MOMENTUM =====
    print("\n" + "="*60)
    print("Optimizer: SGD with Nesterov Momentum")
    print("="*60)
    optimizer = optim.SGD(
        model.parameters(),
        lr=LEARNING_RATE,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
        nesterov=NESTEROV
    )

    print(f"  Learning Rate: {LEARNING_RATE}")
    print(f"  Momentum: {MOMENTUM}")
    print(f"  Weight Decay: {WEIGHT_DECAY}")
    print(f"  Nesterov: {NESTEROV}")

    # Cosine Annealing with Warm Restarts
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=20,      # First restart after 20 epochs
        T_mult=2,    # Double period each restart
        eta_min=1e-7
    )

    print(f"\nScheduler: CosineAnnealingWarmRestarts")
    print(f"  T_0: 20 epochs")
    print(f"  T_mult: 2")
    print(f"  Min LR: 1e-7")

    early_stopping = EarlyStopping(patience=PATIENCE, verbose=True)

    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []

    print("\n" + "="*60)
    print("Training with SGD + Nesterov Momentum")
    print("="*60)
    print("Why SGD is better for validation accuracy:")
    print("  ✅ Finds flatter minima (better generalization)")
    print("  ✅ Less prone to overfitting")
    print("  ✅ More robust weight updates")
    print("  ✅ Often achieves +2-3% better validation")
    print("\nThis may train slower but will achieve higher validation accuracy!")
    print("="*60)

    for epoch in range(NUM_EPOCHS):
        print(f'\nEpoch {epoch+1}/{NUM_EPOCHS}')
        print('-' * 60)

        # Train
        train_loss, train_acc = train_epoch_with_mixup(
            model, train_loader, criterion, optimizer, device, use_mixup=USE_MIXUP
        )
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # Validate
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        # Update learning rate
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}')
        print(f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}')
        print(f'Learning Rate: {current_lr:.6f}')

        # Show train-val gap
        gap = train_acc - val_acc
        if gap > 0.10:
            print(f'⚠️  Train-Val gap: {gap:.4f} (high - may be overfitting)')
        elif gap < 0.05:
            print(f'✅ Train-Val gap: {gap:.4f} (good generalization)')
        else:
            print(f'ℹ️  Train-Val gap: {gap:.4f}')

        # Early stopping
        early_stopping(val_loss, model, epoch)

        if early_stopping.early_stop:
            print(f'\n🛑 Early stopping triggered at epoch {epoch+1}')
            print(f'Best model was at epoch {early_stopping.best_epoch+1}')
            break

    # Load best model
    print(f'\nLoading best model from epoch {early_stopping.best_epoch+1}')
    model.load_state_dict(early_stopping.best_model_state)

    # Save best model
    torch.save({
        'epoch': early_stopping.best_epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_loss': early_stopping.best_loss,
        'train_losses': train_losses,
        'train_accs': train_accs,
        'val_losses': val_losses,
        'val_accs': val_accs,
        'hyperparameters': {
            'lr': LEARNING_RATE,
            'momentum': MOMENTUM,
            'weight_decay': WEIGHT_DECAY,
            'nesterov': NESTEROV,
        }
    }, MODEL_SAVE_PATH)
    print(f'Best model saved to {MODEL_SAVE_PATH}')

    # Plot training history
    plot_training_history(train_losses, train_accs, val_losses, val_accs)

    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"Best Validation Accuracy: {max(val_accs):.4f} ({max(val_accs)*100:.2f}%)")
    print(f"Best Epoch: {early_stopping.best_epoch+1}")
    print(f"Model saved to: {MODEL_SAVE_PATH}")
    print("\nSGD Benefits:")
    print("  ✅ Better generalization to unseen data")
    print("  ✅ More robust weights")
    print("  ✅ Expected +2-3% over Adam baseline")
    print("="*60)


if __name__ == '__main__':
    main()
