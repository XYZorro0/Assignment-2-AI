"""
CNN Design Challenge - IMPROVED Training Script
Optimized for higher accuracy (targeting 80%+ validation)

Key Improvements:
1. Enhanced data augmentation
2. Label smoothing loss
3. Mixup augmentation
4. Optimized hyperparameters
5. Better learning rate scheduling
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


class TinyImageNetDataset(Dataset):
    """Custom Dataset for loading Tiny ImageNet from pickle files"""

    def __init__(self, pickle_file, transform=None):
        print(f"Loading data from {pickle_file}...")
        with open(pickle_file, 'rb') as f:
            data = pickle.load(f)

        if isinstance(data, dict):
            self.images = data.get('images', data.get('data', None))
            self.labels = data.get('labels', data.get('targets', None))
        elif isinstance(data, tuple):
            self.images, self.labels = data
        else:
            raise ValueError("Unknown pickle file structure")

        self.images = np.array(self.images)
        self.labels = np.array(self.labels)

        if self.images.ndim == 4:
            if self.images.shape[-1] == 3:
                pass
            elif self.images.shape[1] == 3:
                self.images = np.transpose(self.images, (0, 2, 3, 1))

        if self.images.max() > 1.0:
            self.images = self.images.astype(np.float32) / 255.0

        self.transform = transform

        print(f"Loaded {len(self.images)} images with shape {self.images.shape}")
        print(f"Labels shape: {self.labels.shape}, Unique classes: {len(np.unique(self.labels))}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            from PIL import Image
            image = Image.fromarray(image)
            image = self.transform(image)
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float()

        return image, label


class ImprovedCNN(nn.Module):
    """
    Improved CNN Architecture with better capacity

    Improvements over baseline:
    - Wider channels (96, 192, 384, 768, 768)
    - Double conv layers in early blocks
    - Optimized dropout placement
    """

    def __init__(self, num_classes=15, dropout_rate=0.4):
        super(ImprovedCNN, self).__init__()

        # Block 1: 64x64x3 -> 32x32x96 (Double conv)
        self.conv1a = nn.Conv2d(3, 96, kernel_size=3, padding=1)
        self.bn1a = nn.BatchNorm2d(96)
        self.conv1b = nn.Conv2d(96, 96, kernel_size=3, padding=1)
        self.bn1b = nn.BatchNorm2d(96)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 2: 32x32x96 -> 16x16x192 (Double conv)
        self.conv2a = nn.Conv2d(96, 192, kernel_size=3, padding=1)
        self.bn2a = nn.BatchNorm2d(192)
        self.conv2b = nn.Conv2d(192, 192, kernel_size=3, padding=1)
        self.bn2b = nn.BatchNorm2d(192)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 3: 16x16x192 -> 8x8x384
        self.conv3a = nn.Conv2d(192, 384, kernel_size=3, padding=1)
        self.bn3a = nn.BatchNorm2d(384)
        self.conv3b = nn.Conv2d(384, 384, kernel_size=3, padding=1)
        self.bn3b = nn.BatchNorm2d(384)
        self.relu3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 4: 8x8x384 -> 4x4x512
        self.conv4a = nn.Conv2d(384, 512, kernel_size=3, padding=1)
        self.bn4a = nn.BatchNorm2d(512)
        self.conv4b = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn4b = nn.BatchNorm2d(512)
        self.relu4 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 5: 4x4x512 -> 2x2x512
        self.conv5a = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn5a = nn.BatchNorm2d(512)
        self.conv5b = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn5b = nn.BatchNorm2d(512)
        self.relu5 = nn.ReLU(inplace=True)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.dropout1 = nn.Dropout(dropout_rate * 0.75)  # 0.3 if dropout_rate=0.4
        self.flatten = nn.Flatten()

        # FC layers: 2x2x512 = 2048
        self.fc1 = nn.Linear(2048, 1024)
        self.bn6 = nn.BatchNorm1d(1024)
        self.relu6 = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout(dropout_rate)  # 0.4

        self.fc2 = nn.Linear(1024, 512)
        self.bn7 = nn.BatchNorm1d(512)
        self.relu7 = nn.ReLU(inplace=True)
        self.dropout3 = nn.Dropout(dropout_rate * 0.5)  # 0.2

        self.fc3 = nn.Linear(512, num_classes)

    def forward(self, x):
        # Block 1
        x = self.relu1(self.bn1a(self.conv1a(x)))
        x = self.relu1(self.bn1b(self.conv1b(x)))
        x = self.pool1(x)

        # Block 2
        x = self.relu2(self.bn2a(self.conv2a(x)))
        x = self.relu2(self.bn2b(self.conv2b(x)))
        x = self.pool2(x)

        # Block 3
        x = self.relu3(self.bn3a(self.conv3a(x)))
        x = self.relu3(self.bn3b(self.conv3b(x)))
        x = self.pool3(x)

        # Block 4
        x = self.relu4(self.bn4a(self.conv4a(x)))
        x = self.relu4(self.bn4b(self.conv4b(x)))
        x = self.pool4(x)

        # Block 5
        x = self.relu5(self.bn5a(self.conv5a(x)))
        x = self.relu5(self.bn5b(self.conv5b(x)))
        x = self.pool5(x)

        # FC layers
        x = self.flatten(x)
        x = self.dropout1(x)
        x = self.dropout2(self.relu6(self.bn6(self.fc1(x))))
        x = self.dropout3(self.relu7(self.bn7(self.fc2(x))))
        x = self.fc3(x)

        return x


class LabelSmoothingLoss(nn.Module):
    """Label smoothing loss to improve generalization"""

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


class EarlyStopping:
    """Early stopping"""

    def __init__(self, patience=10, min_delta=0.0001, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model_state = None
        self.best_epoch = 0

    def __call__(self, val_loss, model, epoch):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model_state = model.state_dict().copy()
            self.best_epoch = epoch
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            if self.verbose:
                print(f'Validation loss improved from {self.best_loss:.6f} to {val_loss:.6f}')
            self.best_loss = val_loss
            self.best_model_state = model.state_dict().copy()
            self.best_epoch = epoch
            self.counter = 0


def train_epoch_with_mixup(model, dataloader, criterion, optimizer, device, use_mixup=True):
    """Train for one epoch with mixup"""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    for images, labels in tqdm(dataloader, desc='Training', leave=False):
        images, labels = images.to(device), labels.to(device)

        if use_mixup and random.random() > 0.5:  # Apply mixup 50% of the time
            images, targets_a, targets_b, lam = mixup_data(images, labels, alpha=0.2, device=device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
        else:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)

        loss.backward()
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


def plot_training_history(train_losses, train_accs, val_losses, val_accs, save_path='training_history_improved.png'):
    """Plot training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(train_losses, label='Train Loss', marker='o')
    ax1.plot(val_losses, label='Val Loss', marker='s')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(train_accs, label='Train Acc', marker='o')
    ax2.plot(val_accs, label='Val Acc', marker='s')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f'Training history plot saved to {save_path}')
    plt.close()


def main():
    # OPTIMIZED Hyperparameters
    BATCH_SIZE = 64
    NUM_EPOCHS = 150  # Increased from 100
    LEARNING_RATE = 0.0005  # Reduced from 0.001 for better convergence
    WEIGHT_DECAY = 1e-4
    DROPOUT_RATE = 0.4  # Reduced from 0.5
    PATIENCE = 20  # Increased from 15
    NUM_CLASSES = 15
    USE_MIXUP = True
    LABEL_SMOOTHING = 0.1

    TRAIN_PICKLE = 'train-70_.pkl'
    VAL_PICKLE = 'validation-10_.pkl'
    MODEL_SAVE_PATH = 'model_improved.pth'

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
        transforms.RandomRotation(20),  # Increased from 15
        transforms.RandomAffine(
            degrees=0,
            translate=(0.15, 0.15),  # Increased from 0.1
            scale=(0.9, 1.1),  # NEW
            shear=10  # NEW
        ),
        transforms.ColorJitter(
            brightness=0.3,  # Increased from 0.2
            contrast=0.3,
            saturation=0.3,
            hue=0.15
        ),
        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=3)  # NEW
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

    # Create improved model
    print("\n" + "="*60)
    print("Creating Improved Model")
    print("="*60)
    model = ImprovedCNN(num_classes=NUM_CLASSES, dropout_rate=DROPOUT_RATE)
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Label smoothing loss
    criterion = LabelSmoothingLoss(classes=NUM_CLASSES, smoothing=LABEL_SMOOTHING)

    # Optimizer - AdamW is better than Adam
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # Cosine annealing scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-7
    )

    early_stopping = EarlyStopping(patience=PATIENCE, verbose=True)

    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []

    print("\n" + "="*60)
    print("Training Improved Model")
    print("="*60)
    print(f"Improvements:")
    print(f"  - Enhanced data augmentation")
    print(f"  - Label smoothing (α={LABEL_SMOOTHING})")
    print(f"  - Mixup augmentation (enabled={USE_MIXUP})")
    print(f"  - Improved architecture (double conv blocks)")
    print(f"  - Optimized hyperparameters")
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

        # Early stopping
        early_stopping(val_loss, model, epoch)

        if early_stopping.early_stop:
            print(f'\nEarly stopping triggered at epoch {epoch+1}')
            print(f'Best model was at epoch {early_stopping.best_epoch+1}')
            break

    # Load best model
    print(f'\nLoading best model from epoch {early_stopping.best_epoch+1}')
    model.load_state_dict(early_stopping.best_model_state)

    # Save best model
    torch.save({
        'epoch': early_stopping.best_epoch,
        'model_state_dict': model.state_dict(),
        'best_val_loss': early_stopping.best_loss,
        'train_losses': train_losses,
        'train_accs': train_accs,
        'val_losses': val_losses,
        'val_accs': val_accs,
    }, MODEL_SAVE_PATH)
    print(f'Best model saved to {MODEL_SAVE_PATH}')

    # Plot training history
    plot_training_history(train_losses, train_accs, val_losses, val_accs)

    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"Best Validation Accuracy: {max(val_accs):.4f}")
    print(f"Model saved to: {MODEL_SAVE_PATH}")
    print("\nExpected improvement: +2-4% over baseline")
    print("="*60)


if __name__ == '__main__':
    main()
