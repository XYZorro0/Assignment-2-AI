"""
Ensemble Prediction Script
Combines predictions from multiple trained models for higher accuracy

Usage:
    python ensemble_predict.py model1.pth model2.pth model3.pth
"""

import sys
import torch
import numpy as np
from submission_model import CustomCNN, predict, TestDataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms


def load_multiple_models(model_paths, device=None):
    """Load multiple models from checkpoint files"""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    models = []

    for path in model_paths:
        print(f"Loading model from {path}...")
        model = CustomCNN(num_classes=15, dropout_rate=0.5)

        checkpoint = torch.load(path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)

        model = model.to(device)
        model.eval()
        models.append(model)

    print(f"✓ Loaded {len(models)} models successfully\n")
    return models


def ensemble_predict_voting(models, test_data, batch_size=64, device=None):
    """
    Ensemble prediction using majority voting

    Args:
        models: List of trained models
        test_data: Test images
        batch_size: Batch size for inference
        device: Device to run inference on

    Returns:
        predictions: Ensemble predictions
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("Making predictions with each model...")

    # Get predictions from each model
    all_predictions = []

    for i, model in enumerate(models, 1):
        print(f"  Model {i}/{len(models)}...", end=' ')
        preds = predict(model, test_data, batch_size=batch_size, device=device)
        all_predictions.append(preds)
        print("✓")

    # Convert to numpy array: (num_models, num_samples)
    all_predictions = np.array(all_predictions)

    print("\nCombining predictions using majority voting...")

    # Majority voting
    final_predictions = []
    for i in range(all_predictions.shape[1]):
        votes = all_predictions[:, i]
        # Get most common prediction
        final_predictions.append(np.bincount(votes).argmax())

    final_predictions = np.array(final_predictions)

    print(f"✓ Ensemble predictions complete: {final_predictions.shape}")

    return final_predictions


def ensemble_predict_averaging(models, test_data, batch_size=64, device=None):
    """
    Ensemble prediction using probability averaging (often better than voting)

    Args:
        models: List of trained models
        test_data: Test images
        batch_size: Batch size for inference
        device: Device to run inference on

    Returns:
        predictions: Ensemble predictions
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Handle different input formats
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

    # Create dataset and dataloader
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_dataset = TestDataset(test_images, transform=test_transform)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False
    )

    print("Making predictions with probability averaging...")

    # Collect probabilities from all models
    all_probabilities = []

    for i, model in enumerate(models, 1):
        print(f"  Model {i}/{len(models)}...", end=' ')
        model.eval()
        probabilities = []

        with torch.no_grad():
            for images in test_loader:
                images = images.to(device)
                outputs = model(images)
                probs = torch.softmax(outputs, dim=1)
                probabilities.extend(probs.cpu().numpy())

        all_probabilities.append(np.array(probabilities))
        print("✓")

    # Average probabilities: (num_models, num_samples, num_classes)
    all_probabilities = np.array(all_probabilities)
    avg_probabilities = np.mean(all_probabilities, axis=0)

    # Get predictions from averaged probabilities
    predictions = np.argmax(avg_probabilities, axis=1)

    print(f"✓ Ensemble predictions complete: {predictions.shape}\n")

    return predictions


def test_ensemble_on_validation(model_paths):
    """Test ensemble on validation set"""
    from sklearn.metrics import accuracy_score, classification_report
    import pickle

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load models
    models = load_multiple_models(model_paths, device)

    # Load validation data
    VAL_PICKLE = 'validation-10_.pkl'

    if not os.path.exists(VAL_PICKLE):
        print(f"Error: {VAL_PICKLE} not found!")
        return

    print(f"Loading validation data from {VAL_PICKLE}...")
    with open(VAL_PICKLE, 'rb') as f:
        data = pickle.load(f)

    if isinstance(data, dict):
        val_images = data.get('images', data.get('data', None))
        val_labels = data.get('labels', data.get('targets', None))
    elif isinstance(data, tuple):
        val_images, val_labels = data

    print(f"Validation set: {len(val_images)} images\n")

    # Test both methods
    print("="*70)
    print("Method 1: Majority Voting")
    print("="*70)
    preds_voting = ensemble_predict_voting(models, val_images, device=device)
    acc_voting = accuracy_score(val_labels, preds_voting)
    print(f"\n✓ Ensemble Accuracy (Voting): {acc_voting:.4f} ({acc_voting*100:.2f}%)")

    print("\n" + "="*70)
    print("Method 2: Probability Averaging")
    print("="*70)
    preds_averaging = ensemble_predict_averaging(models, val_images, device=device)
    acc_averaging = accuracy_score(val_labels, preds_averaging)
    print(f"\n✓ Ensemble Accuracy (Averaging): {acc_averaging:.4f} ({acc_averaging*100:.2f}%)")

    # Compare individual models
    print("\n" + "="*70)
    print("Individual Model Performance:")
    print("="*70)

    for i, model in enumerate(models, 1):
        preds = predict(model, val_images, device=device)
        acc = accuracy_score(val_labels, preds)
        print(f"Model {i}: {acc:.4f} ({acc*100:.2f}%)")

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Best method: {'Averaging' if acc_averaging > acc_voting else 'Voting'}")
    print(f"Best accuracy: {max(acc_voting, acc_averaging):.4f}")
    print("="*70)


def main():
    import os

    if len(sys.argv) < 3:
        print("Usage: python ensemble_predict.py model1.pth model2.pth [model3.pth ...]")
        print("\nExample:")
        print("  python ensemble_predict.py model_ensemble_1.pth model_ensemble_2.pth model_ensemble_3.pth")
        print("\nTo test on validation set:")
        print("  (This will automatically happen if you just provide model paths)")
        sys.exit(1)

    model_paths = sys.argv[1:]

    # Verify all models exist
    for path in model_paths:
        if not os.path.exists(path):
            print(f"Error: Model file not found: {path}")
            sys.exit(1)

    print("="*70)
    print("ENSEMBLE PREDICTION")
    print("="*70)
    print(f"Models to ensemble: {len(model_paths)}")
    for i, path in enumerate(model_paths, 1):
        print(f"  {i}. {path}")
    print("="*70 + "\n")

    # Test on validation set
    test_ensemble_on_validation(model_paths)


if __name__ == '__main__':
    main()
