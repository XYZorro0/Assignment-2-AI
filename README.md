# CNN Design Challenge - Tiny ImageNet Classification

A custom Convolutional Neural Network (CNN) implementation for classifying images from the Tiny ImageNet dataset. This project is designed as a mini AI competition for image classification.

## 🚀 Quick Start Options

### Option 1: Google Colab (Recommended - FREE GPU!)

**Fastest way to get started:**

1. Open `CNN_Training_Notebook.ipynb` in [Google Colab](https://colab.research.google.com/)
2. Enable GPU: `Runtime` → `Change runtime type` → `GPU`
3. Upload your data files (train-70_.pkl, validation-10_.pkl)
4. Run all cells
5. Download `model.pth`

**See [COLAB_GUIDE.md](COLAB_GUIDE.md) for detailed instructions**

### Option 2: Local Machine (Requires Python setup)

Use the Python scripts (.py files) if you have a local environment with GPU.

See sections below for detailed local setup instructions.

---

## 📋 Project Overview

- **Task**: Multi-class image classification
- **Dataset**: Tiny ImageNet (Subset)
- **Classes**: 15
- **Image Size**: 64×64 pixels (RGB)
- **Total Images**: 8,250
  - Training: 70% (5,775 images)
  - Validation: 10% (825 images)
  - Hidden Test: 20% (1,650 images - held by instructor)

## 🏗️ Model Architecture

Custom Sequential CNN with the following structure:

### Convolutional Blocks
1. **Block 1**: Conv2D(3→64) → BatchNorm → ReLU → MaxPool → 32×32×64
2. **Block 2**: Conv2D(64→128) → BatchNorm → ReLU → MaxPool → 16×16×128
3. **Block 3**: Conv2D(128→256) → BatchNorm → ReLU → MaxPool → 8×8×256
4. **Block 4**: Conv2D(256→512) → BatchNorm → ReLU → MaxPool → 4×4×512
5. **Block 5**: Conv2D(512→512) → BatchNorm → ReLU → MaxPool → 2×2×512

### Fully Connected Layers
- Flatten → 2,048 features
- FC1: 2,048 → 1,024 (with BatchNorm, ReLU, Dropout)
- FC2: 1,024 → 512 (with BatchNorm, ReLU, Dropout)
- Output: 512 → 15 classes

**Total Parameters**: ~11M trainable parameters

### Key Features
- ✅ BatchNormalization for training stability
- ✅ Dropout (0.5) for regularization
- ✅ Data augmentation (rotation, flipping, color jitter)
- ✅ Early stopping with patience
- ✅ Learning rate scheduling
- ✅ No pre-trained models or residual connections (as per requirements)

## 📁 Project Structure

```
Assignment-2-AI/
├── README.md                      # This file
├── COLAB_GUIDE.md                 # Complete Google Colab setup guide
├── QUICKSTART.md                  # Quick reference guide
├── requirements.txt               # Python dependencies
│
├── Google Colab Notebooks (Recommended):
│   ├── CNN_Training_Notebook.ipynb     # Complete training pipeline for Colab
│   └── submission_model.ipynb          # Submission file (SUBMIT THIS + model.pth)
│
├── Python Scripts (Local training):
│   ├── train_model.py                  # Training script
│   ├── submission_model.py             # Submission-ready model (alternative to .ipynb)
│   ├── test_submission.py              # Test your submission before submitting
│   └── explore_data.py                 # Data exploration and visualization
│
├── Data Files (Download separately):
│   ├── train-70_.pkl                   # Training data (not in git - too large)
│   └── validation-10_.pkl              # Validation data (not in git - too large)
│
└── Generated Files (after training):
    └── model.pth                       # Trained model weights
```

## 🚀 Quick Start (Local Python)

> **Note:** For Google Colab (easier, FREE GPU), see [COLAB_GUIDE.md](COLAB_GUIDE.md)

### 1. Setup Environment

```bash
# Install dependencies
pip install -r requirements.txt
```

**Required packages:**
- PyTorch >= 2.0.0
- torchvision >= 0.15.0
- numpy >= 1.24.0
- matplotlib >= 3.7.0
- scikit-learn >= 1.3.0
- tqdm >= 4.65.0
- Pillow >= 10.0.0

### 2. Prepare Data

Download the dataset files and place them in the project root:
- `train-70_.pkl` (training data)
- `validation-10_.pkl` (validation data)

### 3. Explore Data (Optional but Recommended)

```bash
python explore_data.py
```

This will:
- Show dataset statistics
- Visualize sample images
- Display class distribution
- Generate plots: `train_samples.png`, `val_samples.png`, `class_distribution.png`

### 4. Train Model

```bash
python train_model.py
```

**Training Configuration:**
- Batch Size: 64
- Max Epochs: 100
- Learning Rate: 0.001 (with ReduceLROnPlateau)
- Weight Decay: 1e-4
- Dropout Rate: 0.5
- Early Stopping Patience: 15 epochs
- Random Seed: 42 (for reproducibility)

**Expected Training Time:**
- With GPU: ~10-20 minutes per epoch (depends on GPU)
- With CPU: ~2-4 hours per epoch (not recommended)

**Outputs:**
- `model.pth` - Trained model weights (REQUIRED FOR SUBMISSION)
- `training_history.png` - Training/validation loss and accuracy plots

### 5. Test Submission

Before submitting, verify your model works correctly:

```bash
python test_submission.py
```

This simulates how the instructor will test your model. It will:
- ✓ Load your model from `model.pth`
- ✓ Test prediction functions
- ✓ Evaluate on validation set
- ✓ Generate confusion matrix
- ✓ Display accuracy metrics

## 📤 Submission Requirements

### Files to Submit

Create a zip file named `GroupX_Assignment.zip` (replace X with your group number) containing:

**Choose ONE of these options:**

**Option A: Jupyter Notebook (Recommended if using Colab)**
1. **submission_model.ipynb** - The submission notebook
2. **model.pth** - Trained model weights

**Option B: Python Script (For local development)**
1. **submission_model.py** - The submission script
2. **model.pth** - Trained model weights

Both options must include:
- Model class definition
- `load_model()` function
- `predict()` function

### Submission Checklist

- [ ] Training completed successfully
- [ ] Validation accuracy is reasonable (>50%)
- [ ] `model.pth` file size is ~40-50 MB
- [ ] Submission file (`submission_model.ipynb` or `.py`) has all required functions
- [ ] Tested that model loads and predicts correctly
- [ ] Created `GroupX_Assignment.zip` with both files
- [ ] Add comment with teammate names and IDs after submission

### How the Instructor Will Test

```python
# 1. Import your model
from submission_model import load_model, predict

# 2. Load your trained model
model = load_model('model.pth')

# 3. Make predictions on hidden test set
predictions = predict(model, test_data)

# 4. Calculate accuracy
accuracy = calculate_accuracy(predictions, true_labels)
```

## 🎯 Model Performance

Expected performance metrics (on validation set):

| Metric | Target |
|--------|--------|
| Validation Accuracy | > 50% |
| Training-Validation Gap | < 10% |
| Convergence | Within 50 epochs |

**Note**: Actual performance depends on hyperparameter tuning and training duration.

## 🔧 Hyperparameter Tuning Tips

To improve performance, consider adjusting:

1. **Architecture**:
   - Add more convolutional layers
   - Experiment with filter sizes
   - Try different pooling strategies

2. **Regularization**:
   - Adjust dropout rates (try 0.3-0.6)
   - Tune weight decay (try 1e-5 to 1e-3)

3. **Training**:
   - Modify learning rate (try 0.0001-0.01)
   - Adjust batch size (32, 64, 128)
   - Extend training epochs

4. **Data Augmentation**:
   - Modify augmentation parameters in `train_model.py`
   - Add more augmentation techniques
   - Adjust augmentation probability

## 📊 Understanding the Output

### Training Output

```
Epoch 1/100
------------------------------------------------------------
Training: 100%
Validation: 100%
Train Loss: 2.1234 | Train Acc: 0.3456
Val Loss: 2.0123 | Val Acc: 0.3789
Learning Rate: 0.001000

EarlyStopping counter: 0/15
```

### Training Plots

`training_history.png` shows:
- Left: Training and validation loss over epochs
- Right: Training and validation accuracy over epochs

**What to look for:**
- ✅ Both losses decreasing
- ✅ Small gap between train and validation accuracy
- ❌ Large gap = overfitting (add more dropout/regularization)
- ❌ Both accuracies low = underfitting (train longer/add capacity)

## 🚫 Important Constraints

As per assignment requirements, you **CANNOT** use:
- ❌ Pre-trained models (ResNet, VGG, EfficientNet, etc.)
- ❌ Transformer or attention mechanisms
- ❌ Residual/skip connections or DenseBlocks
- ❌ Hyperparameter optimization tools (Optuna, Ray Tune)
- ❌ Neural Architecture Search (NAS)

You **CAN** use:
- ✅ Conv2D, Pooling, BatchNorm, Dropout, Flatten, Dense
- ✅ Data augmentation
- ✅ Learning rate scheduling
- ✅ Early stopping
- ✅ Manual hyperparameter tuning

## 🐛 Troubleshooting

### Common Issues

1. **Out of Memory Error**
   ```bash
   # Reduce batch size in train_model.py
   BATCH_SIZE = 32  # or even 16
   ```

2. **Model not loading**
   ```bash
   # Check if model.pth exists
   ls -lh model.pth

   # Verify model architecture matches
   python test_submission.py
   ```

3. **Low accuracy**
   - Train for more epochs
   - Adjust learning rate
   - Add more data augmentation
   - Increase model capacity

4. **Overfitting (train acc >> val acc)**
   - Increase dropout rate
   - Add more data augmentation
   - Reduce model complexity
   - Increase weight decay

5. **Data file not found**
   ```bash
   # Make sure pickle files are in the correct location
   ls -lh *.pkl
   ```

## 🎓 Tips for Better Performance

1. **Start Small**: Begin with the current architecture, verify it works
2. **Monitor Overfitting**: Keep training-validation gap small
3. **Data Augmentation**: Experiment with different augmentation strategies
4. **Learning Rate**: Try different learning rates (0.0001, 0.001, 0.01)
5. **Patience**: Train for enough epochs, use early stopping
6. **Ensemble**: Train multiple models with different seeds (if time permits)
7. **Test Early**: Run `test_submission.py` frequently during development

## 📝 Code Examples

### Loading and Using the Model

```python
from submission_model import load_model, predict
import numpy as np

# Load trained model
model = load_model('model.pth')

# Prepare test data
test_images = np.random.rand(10, 64, 64, 3)  # Example: 10 images

# Make predictions
predictions = predict(model, test_images)
print(f"Predictions: {predictions}")
```

### Custom Prediction with Probabilities

```python
from submission_model import load_model, predict_with_probabilities

model = load_model('model.pth')
predictions, probabilities = predict_with_probabilities(model, test_images)

# Show top predictions
for i in range(len(predictions)):
    top_classes = np.argsort(probabilities[i])[-3:][::-1]
    print(f"Image {i}: Predicted={predictions[i]}")
    for c in top_classes:
        print(f"  Class {c}: {probabilities[i][c]:.4f}")
```

## 📈 Expected Timeline

- **Day 1**: Setup environment, explore data
- **Day 2-3**: Initial training, baseline model
- **Day 4-5**: Hyperparameter tuning, improvements
- **Day 6**: Final training, testing, submission preparation
- **Day 7**: Buffer for issues, final submission

## 🏆 Competition Bonus

Top 3 groups with highest accuracy on the hidden test set will receive bonus marks!

## 📞 Support

If you encounter issues:
1. Check this README thoroughly
2. Run `test_submission.py` to diagnose problems
3. Review the PDF assignment document
4. Ask your instructor or TA

## 📄 License

This project is created for educational purposes as part of an AI course assignment.

---

**Good luck with your submission! 🚀**

Remember: The goal is to build a model that generalizes well to unseen data, not just to achieve high training accuracy.
