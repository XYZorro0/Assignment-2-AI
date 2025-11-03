"""
CNN Design Challenge - Submission Model File
This file contains the model architecture and required functions for submission

Required Components:
1. Model class - The exact neural network architecture
2. Load function - Code to load weights from model.pth
3. Predict function - Takes test data, returns predictions
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


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


def load_model(model_path='model.pth', device=None):
    """
    Load the trained model from checkpoint file

    Args:
        model_path: Path to the model.pth file
        device: Device to load the model on (cpu or cuda)

    Returns:
        model: Loaded model ready for inference
    """
    # Set device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create model instance
    model = CustomCNN(num_classes=15, dropout_rate=0.5)

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)

    # Load model weights
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    # Move model to device and set to evaluation mode
    model = model.to(device)
    model.eval()

    print(f"Model loaded successfully from {model_path}")
    print(f"Device: {device}")

    return model


class TestDataset(Dataset):
    """Dataset wrapper for test data"""

    def __init__(self, images, transform=None):
        """
        Args:
            images: numpy array of images (N, H, W, C) or (N, C, H, W)
            transform: Optional transform to be applied
        """
        self.images = np.array(images)

        # Ensure images are in correct shape: (N, H, W, C)
        if self.images.ndim == 4:
            # If shape is (N, C, H, W), transpose to (N, H, W, C)
            if self.images.shape[1] == 3:
                self.images = np.transpose(self.images, (0, 2, 3, 1))

        # Normalize pixel values to [0, 1] if needed
        if self.images.max() > 1.0:
            self.images = self.images.astype(np.float32) / 255.0

        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]

        # Convert to PIL Image for transforms
        if self.transform:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            from PIL import Image
            image = Image.fromarray(image)
            image = self.transform(image)
        else:
            # Convert to torch tensor
            image = torch.from_numpy(image).permute(2, 0, 1).float()

        return image


def predict(model, test_data, batch_size=64, device=None):
    """
    Make predictions on test data

    Args:
        model: Trained model (already loaded)
        test_data: Test images as numpy array (N, H, W, C) or (N, C, H, W)
                   Can also be a pickle file path or dict with 'images' key
        batch_size: Batch size for inference
        device: Device to run inference on

    Returns:
        predictions: numpy array of predicted class labels (N,)
    """
    # Set device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Ensure model is on correct device and in eval mode
    model = model.to(device)
    model.eval()

    # Handle different input formats
    if isinstance(test_data, str):
        # If string, assume it's a pickle file path
        import pickle
        with open(test_data, 'rb') as f:
            data = pickle.load(f)
        if isinstance(data, dict):
            test_images = data.get('images', data.get('data', None))
        elif isinstance(data, tuple):
            test_images, _ = data
        else:
            test_images = data
    elif isinstance(test_data, dict):
        test_images = test_data.get('images', test_data.get('data', None))
    elif isinstance(test_data, tuple):
        test_images, _ = test_data
    else:
        test_images = test_data

    # Define test transform (same as validation, no augmentation)
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create dataset and dataloader
    test_dataset = TestDataset(test_images, transform=test_transform)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False
    )

    # Make predictions
    predictions = []

    with torch.no_grad():
        for images in test_loader:
            images = images.to(device)

            # Forward pass
            outputs = model(images)

            # Get predicted class
            _, preds = torch.max(outputs, 1)
            predictions.extend(preds.cpu().numpy())

    predictions = np.array(predictions)

    print(f"Predictions shape: {predictions.shape}")

    return predictions


def predict_with_probabilities(model, test_data, batch_size=64, device=None):
    """
    Make predictions on test data and return class probabilities

    Args:
        model: Trained model (already loaded)
        test_data: Test images (same format as predict function)
        batch_size: Batch size for inference
        device: Device to run inference on

    Returns:
        predictions: numpy array of predicted class labels (N,)
        probabilities: numpy array of class probabilities (N, num_classes)
    """
    # Set device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Ensure model is on correct device and in eval mode
    model = model.to(device)
    model.eval()

    # Handle different input formats (same as predict function)
    if isinstance(test_data, str):
        import pickle
        with open(test_data, 'rb') as f:
            data = pickle.load(f)
        if isinstance(data, dict):
            test_images = data.get('images', data.get('data', None))
        elif isinstance(data, tuple):
            test_images, _ = data
        else:
            test_images = data
    elif isinstance(test_data, dict):
        test_images = test_data.get('images', test_data.get('data', None))
    elif isinstance(test_data, tuple):
        test_images, _ = test_data
    else:
        test_images = test_data

    # Define test transform
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create dataset and dataloader
    test_dataset = TestDataset(test_images, transform=test_transform)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False
    )

    # Make predictions
    predictions = []
    probabilities = []

    with torch.no_grad():
        for images in test_loader:
            images = images.to(device)

            # Forward pass
            outputs = model(images)

            # Get probabilities using softmax
            probs = torch.nn.functional.softmax(outputs, dim=1)

            # Get predicted class
            _, preds = torch.max(outputs, 1)

            predictions.extend(preds.cpu().numpy())
            probabilities.extend(probs.cpu().numpy())

    predictions = np.array(predictions)
    probabilities = np.array(probabilities)

    return predictions, probabilities


# Example usage
if __name__ == '__main__':
    """
    Example of how to use this file for prediction
    """
    import sys

    # Load model
    print("Loading model...")
    model = load_model('model.pth')

    # Example: Load test data from pickle
    if len(sys.argv) > 1:
        test_file = sys.argv[1]
        print(f"\nLoading test data from {test_file}...")

        # Make predictions
        print("Making predictions...")
        predictions = predict(model, test_file)

        print(f"\nPredictions for {len(predictions)} images:")
        print(predictions)

        # Save predictions
        output_file = 'predictions.npy'
        np.save(output_file, predictions)
        print(f"\nPredictions saved to {output_file}")
    else:
        print("\nTo test with data, run:")
        print("python submission_model.py <path_to_test_pickle_file>")
