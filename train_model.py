"""
CNN Design Challenge - Training Script
Trains a custom CNN for Tiny ImageNet classification
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
from sklearn.metrics import accuracy_score, confusion_matrix
import random

# Set random seeds for reproducibility
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
        """
        Args:
            pickle_file: Path to the pickle file
            transform: Optional transform to be applied on images
        """
        print(f"Loading data from {pickle_file}...")
        with open(pickle_file, 'rb') as f:
            data = pickle.load(f)

        # Extract images and labels
        # Assuming pickle structure: {'images': array, 'labels': array}
        # Adjust based on actual pickle structure
        if isinstance(data, dict):
            self.images = data.get('images', data.get('data', None))
            self.labels = data.get('labels', data.get('targets', None))
        elif isinstance(data, tuple):
            self.images, self.labels = data
        else:
            raise ValueError("Unknown pickle file structure")

        # Convert to numpy arrays if needed
        self.images = np.array(self.images)
        self.labels = np.array(self.labels)

        # Ensure images are in correct shape: (N, H, W, C) or (N, C, H, W)
        if self.images.ndim == 4:
            # If shape is (N, H, W, C), keep it
            if self.images.shape[-1] == 3:
                pass  # Already in (N, H, W, C) format
            # If shape is (N, C, H, W), transpose to (N, H, W, C)
            elif self.images.shape[1] == 3:
                self.images = np.transpose(self.images, (0, 2, 3, 1))

        # Normalize pixel values to [0, 1] if needed
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

        # Convert to PIL Image for transforms
        if self.transform:
            # Ensure image is in correct format for transforms
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            from PIL import Image
            image = Image.fromarray(image)
            image = self.transform(image)
        else:
            # Convert to torch tensor
            image = torch.from_numpy(image).permute(2, 0, 1).float()

        return image, label


class CustomCNN(nn.Module):
    """
    Custom CNN Architecture for Tiny ImageNet Classification

    Architecture:
    - 5 Convolutional Blocks (Conv2D -> BatchNorm -> ReLU -> MaxPool)
    - Dropout for regularization
    - 2 Fully Connected layers
    - Output layer with 15 classes

    Input: 64x64x3 RGB images
    Output: 15 classes
    """

    def __init__(self, num_classes=15, dropout_rate=0.5):
        super(CustomCNN, self).__init__()

        # Block 1: 64x64x3 -> 32x32x64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 2: 32x32x64 -> 16x16x128
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 3: 16x16x128 -> 8x8x256
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.relu3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 4: 8x8x256 -> 4x4x512
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.relu4 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 5: 4x4x512 -> 2x2x512
        self.conv5 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(512)
        self.relu5 = nn.ReLU(inplace=True)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Dropout for regularization
        self.dropout1 = nn.Dropout(dropout_rate)

        # Flatten layer
        self.flatten = nn.Flatten()

        # Fully connected layers
        # After all pooling: 2x2x512 = 2048
        self.fc1 = nn.Linear(2048, 1024)
        self.bn6 = nn.BatchNorm1d(1024)
        self.relu6 = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout(dropout_rate)

        self.fc2 = nn.Linear(1024, 512)
        self.bn7 = nn.BatchNorm1d(512)
        self.relu7 = nn.ReLU(inplace=True)
        self.dropout3 = nn.Dropout(dropout_rate / 2)

        # Output layer
        self.fc3 = nn.Linear(512, num_classes)

    def forward(self, x):
        # Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        # Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        # Block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.pool3(x)

        # Block 4
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)
        x = self.pool4(x)

        # Block 5
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu5(x)
        x = self.pool5(x)

        # Flatten
        x = self.flatten(x)
        x = self.dropout1(x)

        # FC layers
        x = self.fc1(x)
        x = self.bn6(x)
        x = self.relu6(x)
        x = self.dropout2(x)

        x = self.fc2(x)
        x = self.bn7(x)
        x = self.relu7(x)
        x = self.dropout3(x)

        # Output
        x = self.fc3(x)

        return x


class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve"""

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


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    for images, labels in tqdm(dataloader, desc='Training', leave=False):
        images, labels = images.to(device), labels.to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Statistics
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

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Statistics
            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)

    return epoch_loss, epoch_acc


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler,
                num_epochs, device, save_path='model.pth', patience=15):
    """Complete training loop with early stopping"""

    early_stopping = EarlyStopping(patience=patience, verbose=True)

    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []

    print(f"\nStarting training on device: {device}")
    print(f"Total epochs: {num_epochs}, Early stopping patience: {patience}\n")

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 60)

        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # Validate
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        # Update learning rate
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}')
        print(f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}')
        print(f'Learning Rate: {current_lr:.6f}\n')

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
    }, save_path)
    print(f'Best model saved to {save_path}')

    return train_losses, train_accs, val_losses, val_accs


def plot_training_history(train_losses, train_accs, val_losses, val_accs, save_path='training_history.png'):
    """Plot training and validation metrics"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss plot
    ax1.plot(train_losses, label='Train Loss', marker='o')
    ax1.plot(val_losses, label='Val Loss', marker='s')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)

    # Accuracy plot
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
    # Hyperparameters
    BATCH_SIZE = 64
    NUM_EPOCHS = 100
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-4
    DROPOUT_RATE = 0.5
    PATIENCE = 15
    NUM_CLASSES = 15

    # Paths to data files
    TRAIN_PICKLE = 'train-70_.pkl'
    VAL_PICKLE = 'validation-10_.pkl'
    MODEL_SAVE_PATH = 'model.pth'

    # Check if data files exist
    if not os.path.exists(TRAIN_PICKLE):
        print(f"Error: {TRAIN_PICKLE} not found!")
        print("Please place the training pickle file in the current directory.")
        return

    if not os.path.exists(VAL_PICKLE):
        print(f"Error: {VAL_PICKLE} not found!")
        print("Please place the validation pickle file in the current directory.")
        return

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Data augmentation for training
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # No augmentation for validation
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

    # Create dataloaders
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

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True, min_lr=1e-7
    )

    # Train model
    print("\n" + "="*60)
    print("Training Model")
    print("="*60)
    train_losses, train_accs, val_losses, val_accs = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=NUM_EPOCHS,
        device=device,
        save_path=MODEL_SAVE_PATH,
        patience=PATIENCE
    )

    # Plot training history
    plot_training_history(train_losses, train_accs, val_losses, val_accs)

    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"Best Validation Accuracy: {max(val_accs):.4f}")
    print(f"Model saved to: {MODEL_SAVE_PATH}")


if __name__ == '__main__':
    main()
