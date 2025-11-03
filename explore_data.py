"""
Data Exploration Script
Helps visualize and understand the Tiny ImageNet dataset
Run this before training to understand your data
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter


def load_and_explore_pickle(pickle_file):
    """Load and explore the structure of pickle file"""
    print(f"\n{'='*70}")
    print(f"Exploring: {pickle_file}")
    print('='*70)

    with open(pickle_file, 'rb') as f:
        data = pickle.load(f)

    # Determine data structure
    print(f"\nData type: {type(data)}")

    if isinstance(data, dict):
        print(f"Dictionary keys: {data.keys()}")
        images = data.get('images', data.get('data', None))
        labels = data.get('labels', data.get('targets', None))
    elif isinstance(data, tuple):
        print(f"Tuple with {len(data)} elements")
        images, labels = data
    else:
        print("Unknown structure")
        return None, None

    # Print shapes and info
    images = np.array(images)
    labels = np.array(labels)

    print(f"\n📊 Dataset Statistics:")
    print(f"  Images shape: {images.shape}")
    print(f"  Labels shape: {labels.shape}")
    print(f"  Image dtype: {images.dtype}")
    print(f"  Label dtype: {labels.dtype}")
    print(f"\n  Pixel value range: [{images.min()}, {images.max()}]")
    print(f"  Labels range: [{labels.min()}, {labels.max()}]")
    print(f"  Number of unique classes: {len(np.unique(labels))}")
    print(f"  Unique classes: {sorted(np.unique(labels))}")

    # Class distribution
    class_counts = Counter(labels)
    print(f"\n📈 Class Distribution:")
    for class_id in sorted(class_counts.keys()):
        print(f"  Class {class_id}: {class_counts[class_id]} images")

    return images, labels


def visualize_samples(images, labels, num_samples=20, save_path='sample_images.png'):
    """Visualize random samples from the dataset"""
    print(f"\n{'='*70}")
    print("Visualizing Sample Images")
    print('='*70)

    # Ensure images are in [0, 1] range for visualization
    if images.max() > 1.0:
        vis_images = images.astype(np.float32) / 255.0
    else:
        vis_images = images

    # Handle different image formats
    if vis_images.ndim == 4:
        # If shape is (N, C, H, W), transpose to (N, H, W, C)
        if vis_images.shape[1] == 3:
            vis_images = np.transpose(vis_images, (0, 2, 3, 1))

    # Select random samples
    num_samples = min(num_samples, len(images))
    indices = np.random.choice(len(images), num_samples, replace=False)

    # Calculate grid size
    cols = 5
    rows = (num_samples + cols - 1) // cols

    # Create figure
    fig, axes = plt.subplots(rows, cols, figsize=(15, 3*rows))
    axes = axes.flatten() if num_samples > 1 else [axes]

    for idx, ax in enumerate(axes):
        if idx < num_samples:
            img_idx = indices[idx]
            ax.imshow(vis_images[img_idx])
            ax.set_title(f'Class: {labels[img_idx]}', fontsize=10)
            ax.axis('off')
        else:
            ax.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Sample images saved to {save_path}")
    plt.close()


def plot_class_distribution(train_labels, val_labels, save_path='class_distribution.png'):
    """Plot class distribution for training and validation sets"""
    print(f"\n{'='*70}")
    print("Plotting Class Distribution")
    print('='*70)

    train_counts = Counter(train_labels)
    val_counts = Counter(val_labels)

    classes = sorted(set(train_labels) | set(val_labels))
    train_vals = [train_counts.get(c, 0) for c in classes]
    val_vals = [val_counts.get(c, 0) for c in classes]

    x = np.arange(len(classes))
    width = 0.35

    fig, ax = plt.subplots(figsize=(14, 6))
    bars1 = ax.bar(x - width/2, train_vals, width, label='Training', alpha=0.8)
    bars2 = ax.bar(x + width/2, val_vals, width, label='Validation', alpha=0.8)

    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('Number of Images', fontsize=12)
    ax.set_title('Class Distribution: Training vs Validation', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Class distribution plot saved to {save_path}")
    plt.close()


def analyze_image_statistics(images, title="Image Statistics"):
    """Analyze pixel value statistics"""
    print(f"\n{'='*70}")
    print(f"{title}")
    print('='*70)

    # Normalize to [0, 1] if needed
    if images.max() > 1.0:
        images = images.astype(np.float32) / 255.0

    # Handle different image formats
    if images.ndim == 4:
        if images.shape[1] == 3:  # (N, C, H, W)
            images = np.transpose(images, (0, 2, 3, 1))  # (N, H, W, C)

    # Calculate statistics per channel
    print("\nPer-channel statistics (RGB):")
    for i, channel in enumerate(['Red', 'Green', 'Blue']):
        channel_data = images[:, :, :, i]
        print(f"\n  {channel} channel:")
        print(f"    Mean: {channel_data.mean():.4f}")
        print(f"    Std:  {channel_data.std():.4f}")
        print(f"    Min:  {channel_data.min():.4f}")
        print(f"    Max:  {channel_data.max():.4f}")


def main():
    """Main exploration function"""
    print("\n" + "="*70)
    print(" "*20 + "DATA EXPLORATION")
    print("="*70)

    # File paths
    TRAIN_PICKLE = 'train-70_.pkl'
    VAL_PICKLE = 'validation-10_.pkl'

    # Load and explore training data
    try:
        train_images, train_labels = load_and_explore_pickle(TRAIN_PICKLE)
        analyze_image_statistics(train_images, "Training Set - Image Statistics")
        visualize_samples(train_images, train_labels, num_samples=20, save_path='train_samples.png')
    except FileNotFoundError:
        print(f"\n⚠ Warning: {TRAIN_PICKLE} not found. Skipping training data exploration.")
        train_images, train_labels = None, None
    except Exception as e:
        print(f"\n❌ Error loading training data: {str(e)}")
        train_images, train_labels = None, None

    # Load and explore validation data
    try:
        val_images, val_labels = load_and_explore_pickle(VAL_PICKLE)
        analyze_image_statistics(val_images, "Validation Set - Image Statistics")
        visualize_samples(val_images, val_labels, num_samples=20, save_path='val_samples.png')
    except FileNotFoundError:
        print(f"\n⚠ Warning: {VAL_PICKLE} not found. Skipping validation data exploration.")
        val_images, val_labels = None, None
    except Exception as e:
        print(f"\n❌ Error loading validation data: {str(e)}")
        val_images, val_labels = None, None

    # Plot class distribution
    if train_labels is not None and val_labels is not None:
        plot_class_distribution(train_labels, val_labels)

    print("\n" + "="*70)
    print(" "*20 + "EXPLORATION COMPLETE")
    print("="*70)
    print("\nGenerated files:")
    print("  - train_samples.png (if training data available)")
    print("  - val_samples.png (if validation data available)")
    print("  - class_distribution.png (if both datasets available)")
    print("\n" + "="*70)


if __name__ == '__main__':
    main()
