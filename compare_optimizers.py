"""
Compare Different Optimizers

This script tests your model with different optimizers to find the best one
for maximum validation accuracy.

Optimizers tested:
1. Adam (baseline)
2. AdamW
3. SGD + Momentum
4. SGD + Nesterov Momentum (recommended)

Usage:
    python compare_optimizers.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
from train_model import CustomCNN, TinyImageNetDataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


def quick_train_test(optimizer_name, model, train_loader, val_loader, criterion, device, num_epochs=30):
    """
    Quick training run to test optimizer

    Args:
        optimizer_name: Name of optimizer
        model: Model to train
        train_loader: Training data
        val_loader: Validation data
        criterion: Loss function
        device: Device to train on
        num_epochs: Number of epochs to train

    Returns:
        best_val_acc: Best validation accuracy achieved
    """

    # Create optimizer based on name
    if optimizer_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    elif optimizer_name == "AdamW":
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    elif optimizer_name == "SGD+Momentum":
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    elif optimizer_name == "SGD+Nesterov":
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4, nesterov=True)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    # Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    val_accs = []
    best_val_acc = 0.0

    print(f"\nTesting {optimizer_name}...")
    print("="*60)

    for epoch in range(num_epochs):
        # Train
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Validate
        model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_acc = accuracy_score(all_labels, all_preds)
        val_accs.append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc

        scheduler.step()

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{num_epochs} - Val Acc: {val_acc:.4f} - Best: {best_val_acc:.4f}")

    print(f"✓ {optimizer_name} completed - Best Val Acc: {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")

    return best_val_acc, val_accs


def main():
    import os

    TRAIN_PICKLE = 'train-70_.pkl'
    VAL_PICKLE = 'validation-10_.pkl'

    if not os.path.exists(TRAIN_PICKLE) or not os.path.exists(VAL_PICKLE):
        print("Error: Data files not found!")
        print("Please ensure train-70_.pkl and validation-10_.pkl are in the current directory.")
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    # Load data
    print("Loading datasets...")

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = TinyImageNetDataset(TRAIN_PICKLE, transform=train_transform)
    val_dataset = TinyImageNetDataset(VAL_PICKLE, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)

    criterion = nn.CrossEntropyLoss()

    # Test each optimizer
    optimizers_to_test = [
        "Adam",
        "AdamW",
        "SGD+Momentum",
        "SGD+Nesterov"
    ]

    results = {}
    all_val_accs = {}

    print("\n" + "="*70)
    print(" "*20 + "OPTIMIZER COMPARISON")
    print("="*70)
    print("Testing each optimizer with 30 epochs of training")
    print("This will take approximately 20-40 minutes depending on GPU")
    print("="*70)

    for opt_name in optimizers_to_test:
        # Create fresh model for each test
        model = CustomCNN(num_classes=15, dropout_rate=0.5)
        model = model.to(device)

        best_acc, val_accs = quick_train_test(
            opt_name, model, train_loader, val_loader, criterion, device, num_epochs=30
        )

        results[opt_name] = best_acc
        all_val_accs[opt_name] = val_accs

    # Print results
    print("\n" + "="*70)
    print(" "*25 + "FINAL RESULTS")
    print("="*70)
    print(f"{'Optimizer':<20} {'Best Val Accuracy':<20} {'Improvement':<15}")
    print("-"*70)

    baseline = results["Adam"]

    sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)

    for opt_name, acc in sorted_results:
        improvement = (acc - baseline) * 100
        marker = "🏆" if acc == max(results.values()) else ""
        print(f"{opt_name:<20} {acc:.4f} ({acc*100:.2f}%)    +{improvement:.2f}%  {marker}")

    print("="*70)

    # Find best optimizer
    best_opt = max(results, key=results.get)
    print(f"\n✅ BEST OPTIMIZER: {best_opt}")
    print(f"   Accuracy: {results[best_opt]:.4f} ({results[best_opt]*100:.2f}%)")
    print(f"   Improvement over Adam: +{(results[best_opt] - baseline)*100:.2f}%")

    # Plot comparison
    fig, ax = plt.subplots(figsize=(12, 6))

    for opt_name, val_accs in all_val_accs.items():
        ax.plot(val_accs, label=opt_name, marker='o' if 'SGD' in opt_name else 's', alpha=0.7)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation Accuracy')
    ax.set_title('Optimizer Comparison - Validation Accuracy Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('optimizer_comparison.png', dpi=300, bbox_inches='tight')
    print(f"\n📊 Comparison plot saved to: optimizer_comparison.png")

    print("\n" + "="*70)
    print("RECOMMENDATIONS:")
    print("="*70)

    if best_opt in ["SGD+Nesterov", "SGD+Momentum"]:
        print("✅ SGD-based optimizers perform best for your model!")
        print("   Use train_sgd_nesterov.py for full training")
        print("   Expected final accuracy with full training: 78-80%+")
    else:
        print("ℹ️  Adam-based optimizers perform best in this quick test")
        print("   However, SGD often improves with longer training")
        print("   Consider running full training with both and comparing")

    print("\nKey Insights:")
    if results["SGD+Nesterov"] > results["Adam"]:
        print(f"  • SGD+Nesterov is +{(results['SGD+Nesterov'] - results['Adam'])*100:.2f}% better than Adam")
        print("  • SGD finds flatter minima → better generalization")
        print("  • This gap usually INCREASES with more training epochs")
    else:
        print("  • Adam converged faster in this short test")
        print("  • Try longer training - SGD often catches up and surpasses")

    print("\nNext Steps:")
    print("  1. Run full training with best optimizer")
    print("  2. Use 150-200 epochs for SGD-based optimizers")
    print("  3. Monitor train-val gap (should be <10%)")
    print("="*70)


if __name__ == '__main__':
    main()
