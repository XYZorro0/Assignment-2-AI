"""
Test Submission Script
Tests that the submission model file works correctly with the validation data
This simulates how the instructor will test your model
"""

import os
import pickle
import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Import from submission file
from submission_model import CustomCNN, load_model, predict, predict_with_probabilities


def load_validation_data(val_pickle='validation-10_.pkl'):
    """Load validation data from pickle file"""
    print(f"Loading validation data from {val_pickle}...")

    with open(val_pickle, 'rb') as f:
        data = pickle.load(f)

    # Extract images and labels
    if isinstance(data, dict):
        images = data.get('images', data.get('data', None))
        labels = data.get('labels', data.get('targets', None))
    elif isinstance(data, tuple):
        images, labels = data
    else:
        raise ValueError("Unknown pickle file structure")

    images = np.array(images)
    labels = np.array(labels)

    print(f"Validation images shape: {images.shape}")
    print(f"Validation labels shape: {labels.shape}")
    print(f"Number of unique classes: {len(np.unique(labels))}")

    return images, labels


def test_model_loading(model_path='model.pth'):
    """Test 1: Check if model loads correctly"""
    print("\n" + "="*60)
    print("TEST 1: Model Loading")
    print("="*60)

    if not os.path.exists(model_path):
        print(f"❌ FAILED: {model_path} not found!")
        return False

    try:
        model = load_model(model_path)
        print("✓ Model loaded successfully")

        # Check if model is in eval mode
        if not model.training:
            print("✓ Model is in evaluation mode")
        else:
            print("⚠ Warning: Model is not in evaluation mode")

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        print(f"✓ Model has {total_params:,} parameters")

        return True
    except Exception as e:
        print(f"❌ FAILED: Error loading model: {str(e)}")
        return False


def test_prediction_function(model, val_images):
    """Test 2: Check if predict function works"""
    print("\n" + "="*60)
    print("TEST 2: Prediction Function")
    print("="*60)

    try:
        # Test with a small subset first
        test_subset = val_images[:10]
        predictions = predict(model, test_subset, batch_size=4)

        print(f"✓ Predictions generated successfully")
        print(f"✓ Predictions shape: {predictions.shape}")
        print(f"✓ Sample predictions: {predictions[:5]}")

        # Check if predictions are valid class indices
        if predictions.min() >= 0 and predictions.max() < 15:
            print(f"✓ Predictions are valid (range: {predictions.min()} to {predictions.max()})")
        else:
            print(f"⚠ Warning: Predictions outside expected range (0-14)")

        return True
    except Exception as e:
        print(f"❌ FAILED: Error in prediction: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_full_validation(model, val_images, val_labels):
    """Test 3: Run full validation and compute accuracy"""
    print("\n" + "="*60)
    print("TEST 3: Full Validation Set Evaluation")
    print("="*60)

    try:
        # Make predictions on full validation set
        print("Making predictions on validation set...")
        predictions = predict(model, val_images, batch_size=64)

        # Calculate accuracy
        accuracy = accuracy_score(val_labels, predictions)
        print(f"\n✓ Validation Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

        # Per-class accuracy
        print("\n" + "-"*60)
        print("Classification Report:")
        print("-"*60)
        report = classification_report(val_labels, predictions)
        print(report)

        return True, accuracy, predictions
    except Exception as e:
        print(f"❌ FAILED: Error in evaluation: {str(e)}")
        import traceback
        traceback.print_exc()
        return False, 0.0, None


def plot_confusion_matrix(val_labels, predictions, save_path='confusion_matrix.png'):
    """Plot confusion matrix"""
    print("\n" + "="*60)
    print("Generating Confusion Matrix")
    print("="*60)

    try:
        cm = confusion_matrix(val_labels, predictions)

        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True)
        plt.title('Confusion Matrix - Validation Set', fontsize=16)
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Confusion matrix saved to {save_path}")
        plt.close()
    except Exception as e:
        print(f"⚠ Warning: Could not generate confusion matrix: {str(e)}")


def test_probability_predictions(model, val_images):
    """Test 4: Check probability predictions"""
    print("\n" + "="*60)
    print("TEST 4: Probability Predictions")
    print("="*60)

    try:
        # Test with a small subset
        test_subset = val_images[:10]
        predictions, probabilities = predict_with_probabilities(model, test_subset)

        print(f"✓ Probabilities shape: {probabilities.shape}")
        print(f"✓ Probability range: [{probabilities.min():.4f}, {probabilities.max():.4f}]")

        # Check if probabilities sum to 1
        prob_sums = probabilities.sum(axis=1)
        if np.allclose(prob_sums, 1.0):
            print("✓ Probabilities sum to 1.0")
        else:
            print(f"⚠ Warning: Probabilities don't sum to 1.0 (range: {prob_sums.min():.4f} to {prob_sums.max():.4f})")

        # Show top-3 predictions for first sample
        print("\nExample prediction (first sample):")
        top3_idx = np.argsort(probabilities[0])[-3:][::-1]
        for i, idx in enumerate(top3_idx, 1):
            print(f"  Top-{i}: Class {idx} (probability: {probabilities[0][idx]:.4f})")

        return True
    except Exception as e:
        print(f"⚠ Warning: Error in probability predictions: {str(e)}")
        return False


def main():
    """Run all tests"""
    print("\n" + "="*70)
    print(" "*15 + "SUBMISSION MODEL TESTING")
    print("="*70)
    print("\nThis script tests your submission to ensure it will work correctly")
    print("when evaluated by the instructor.\n")

    # Configuration
    MODEL_PATH = 'model.pth'
    VAL_PICKLE = 'validation-10_.pkl'

    # Check if validation data exists
    if not os.path.exists(VAL_PICKLE):
        print(f"❌ Error: {VAL_PICKLE} not found!")
        print("Please place the validation pickle file in the current directory.")
        return

    # Load validation data
    val_images, val_labels = load_validation_data(VAL_PICKLE)

    # Test 1: Model Loading
    if not test_model_loading(MODEL_PATH):
        print("\n❌ Model loading failed. Please fix the issue and try again.")
        return

    # Load model for subsequent tests
    model = load_model(MODEL_PATH)

    # Test 2: Prediction Function
    if not test_prediction_function(model, val_images):
        print("\n❌ Prediction function failed. Please fix the issue and try again.")
        return

    # Test 3: Full Validation
    success, accuracy, predictions = test_full_validation(model, val_images, val_labels)
    if not success:
        print("\n❌ Validation failed. Please fix the issue and try again.")
        return

    # Test 4: Probability Predictions
    test_probability_predictions(model, val_images)

    # Generate confusion matrix
    if predictions is not None:
        plot_confusion_matrix(val_labels, predictions)

    # Final Summary
    print("\n" + "="*70)
    print(" "*20 + "TEST SUMMARY")
    print("="*70)
    print("\n✓ All critical tests passed!")
    print(f"✓ Validation Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print("\nYour submission is ready! Make sure to submit:")
    print("  1. submission_model.py (or .ipynb)")
    print("  2. model.pth")
    print("\n" + "="*70)


if __name__ == '__main__':
    main()
