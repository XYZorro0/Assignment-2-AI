# Google Colab Setup Guide

Complete guide for running the CNN training on Google Colab (FREE GPU!)

## 🚀 Quick Start (5 Minutes)

### Step 1: Open Training Notebook in Colab

1. Go to [Google Colab](https://colab.research.google.com/)
2. Click `File` → `Upload notebook`
3. Upload `CNN_Training_Notebook.ipynb` from this repository

**OR** use this link format:
```
https://colab.research.google.com/github/[YOUR_USERNAME]/Assignment-2-AI/blob/main/CNN_Training_Notebook.ipynb
```

### Step 2: Enable GPU

**IMPORTANT:** You must enable GPU for reasonable training speed!

1. Click `Runtime` → `Change runtime type`
2. Select `T4 GPU` (or any available GPU)
3. Click `Save`

Verify GPU is enabled by running the first cell - it should show:
```
CUDA available: True
GPU: Tesla T4
```

### Step 3: Upload Data Files

You have two options:

#### Option A: Direct Upload (Simpler, but slower)

1. Click the folder icon 📁 on the left sidebar
2. Click upload button
3. Upload both files:
   - `train-70_.pkl`
   - `validation-10_.pkl`

⚠️ **Note**: Files will be deleted when runtime disconnects!

#### Option B: Google Drive (Recommended)

1. Upload your pickle files to Google Drive first
2. Run this cell in the notebook:

```python
from google.colab import drive
drive.mount('/content/drive')

# Update file paths
TRAIN_PICKLE = '/content/drive/MyDrive/train-70_.pkl'
VAL_PICKLE = '/content/drive/MyDrive/validation-10_.pkl'
```

3. Follow prompts to authorize access

### Step 4: Run Training

1. Click `Runtime` → `Run all` (or Ctrl+F9)
2. Wait for training to complete (30-60 minutes with GPU)
3. Monitor progress in the output

### Step 5: Download Model

After training completes:

1. The notebook will automatically prompt you to download `model.pth`
2. OR manually download:
   - Click folder icon 📁
   - Find `model.pth`
   - Right-click → Download

---

## 📋 Detailed Instructions

### Understanding the Notebooks

**1. CNN_Training_Notebook.ipynb** - Complete training pipeline
- Data exploration and visualization
- Model training with early stopping
- Performance evaluation
- Saves `model.pth`

**2. submission_model.ipynb** - Submission file
- Model architecture
- Load and predict functions
- Testing utilities
- This is what you submit (along with model.pth)

### Training Process

The training notebook will:

1. **Install dependencies** (~30 seconds)
2. **Load data** (~1-2 minutes)
3. **Explore data** (optional, ~30 seconds)
4. **Train model** (~30-60 minutes with GPU)
   - Up to 100 epochs
   - Early stopping (patience=15)
   - Learning rate scheduling
5. **Evaluate** (~30 seconds)
6. **Save model** (~5 seconds)

### Expected Training Time

| Hardware | Time per Epoch | Total Time (50 epochs) |
|----------|----------------|------------------------|
| Colab GPU (T4) | ~1-2 minutes | 50-100 minutes |
| Colab CPU | ~2-4 hours | 100-200 hours ❌ |

**Always use GPU!**

### Monitoring Training

Watch for these metrics:

```
Epoch 1/100
------------------------------------------------------------
Training: 100%|██████████| 90/90
Validation: 100%|██████████| 13/13
Train Loss: 2.1234 | Train Acc: 0.3456
Val Loss: 2.0123 | Val Acc: 0.3789
Learning Rate: 0.001000

Validation loss improved from 2.1000 to 2.0123
```

**Good signs:**
- ✅ Loss decreasing
- ✅ Accuracy increasing
- ✅ Small gap between train and val accuracy

**Bad signs:**
- ❌ Train acc >> Val acc (overfitting)
- ❌ Both accuracies stuck at low values (underfitting)

---

## 🎯 Submission Workflow

### After Training Completes

1. **Download model.pth** (from Colab)
2. **Open submission_model.ipynb** in Colab
3. **Upload model.pth** to Colab
4. **Run all cells** to test
5. **Download submission_model.ipynb**
6. **Create zip file** on your computer:
   ```
   GroupX_Assignment.zip
   ├── submission_model.ipynb
   └── model.pth
   ```
7. **Submit** to your course platform

### Verification Checklist

Before submitting, ensure:

- [ ] `model.pth` file exists and is 40-50 MB
- [ ] `submission_model.ipynb` runs without errors
- [ ] Test section shows: "✓ Model loaded successfully"
- [ ] Predictions are in range [0, 14]
- [ ] Validation accuracy > 50% (if you tested)
- [ ] Both files are in zip: `GroupX_Assignment.zip`

---

## 🔧 Troubleshooting

### Issue: "Runtime disconnected"

**Solution:**
- Colab disconnects after ~30 min of inactivity
- Keep the tab open and check periodically
- Use Google Drive to save progress
- Consider running overnight

### Issue: "Out of Memory"

**Solution 1:** Reduce batch size
```python
# In the hyperparameters cell, change:
BATCH_SIZE = 32  # or even 16
```

**Solution 2:** Restart runtime
```
Runtime → Restart runtime
```

### Issue: "GPU not available"

**Solution:**
1. Check runtime type: `Runtime → Change runtime type`
2. Select GPU
3. If still not available, Colab may have hit usage limits
4. Wait a few hours or try a different Google account

### Issue: "File not found: train-70_.pkl"

**Solution:**
- Make sure you uploaded the file
- Check the file name exactly matches
- If using Google Drive, check the path

### Issue: "Training is very slow"

**Check:**
1. Is GPU enabled? Run this cell:
   ```python
   import torch
   print(torch.cuda.is_available())  # Should be True
   ```
2. If False, enable GPU (see Step 2 above)

### Issue: "Model accuracy is low (<40%)"

**Try:**
1. Train for more epochs (increase `NUM_EPOCHS`)
2. Reduce learning rate: `LEARNING_RATE = 0.0005`
3. Increase data augmentation
4. Train multiple times with different seeds

### Issue: "Can't download model.pth"

**Manual download:**
1. Click folder icon 📁 on left
2. Find `model.pth`
3. Right-click → Download
4. OR use this code:
   ```python
   from google.colab import files
   files.download('model.pth')
   ```

---

## 💡 Pro Tips

### 1. Save Checkpoints to Drive

Add this to save progress to Google Drive:

```python
# After training completes
import shutil
shutil.copy('model.pth', '/content/drive/MyDrive/model.pth')
print("✓ Model backed up to Google Drive")
```

### 2. Monitor Training Remotely

Keep the Colab tab open on your phone/tablet to monitor progress

### 3. Multiple Training Runs

Try different hyperparameters in parallel:

1. Make copies of the notebook
2. Modify hyperparameters in each
3. Run all simultaneously (different Colab sessions)
4. Compare results

### 4. Experiment with Hyperparameters

In the hyperparameters cell, try:

```python
# Conservative (less overfitting)
BATCH_SIZE = 32
LEARNING_RATE = 0.0005
DROPOUT_RATE = 0.6

# Aggressive (more capacity)
BATCH_SIZE = 128
LEARNING_RATE = 0.001
DROPOUT_RATE = 0.4
```

### 5. Early Testing

Don't wait for full training! After 10-15 epochs:
- Check if accuracy is improving
- If stuck, stop and adjust hyperparameters
- Save time by testing early

---

## 📊 Expected Results

### After 30-50 Epochs

| Metric | Expected Range |
|--------|---------------|
| Training Accuracy | 60-80% |
| Validation Accuracy | 50-70% |
| Training Loss | 0.5-1.5 |
| Validation Loss | 1.0-2.0 |
| Model Size | 40-50 MB |
| Training Time (GPU) | 30-90 minutes |

### Performance Indicators

**Excellent** (Top 3 for bonus!)
- Val Accuracy: > 65%
- Train-Val gap: < 5%

**Good**
- Val Accuracy: 55-65%
- Train-Val gap: 5-10%

**Acceptable**
- Val Accuracy: 50-55%
- Train-Val gap: 10-15%

**Needs improvement**
- Val Accuracy: < 50%
- Train-Val gap: > 15%

---

## 🎓 Advanced: Improving Performance

### 1. Data Augmentation

Modify augmentation in the notebook:

```python
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(20),  # Increase from 15
    transforms.RandomAffine(degrees=0, translate=(0.15, 0.15)),  # Increase
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.15),
    transforms.RandomCrop(64, padding=4),  # Add random crop
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```

### 2. Learning Rate Tuning

Try different learning rates:

```python
# Lower LR (more stable, slower)
LEARNING_RATE = 0.0005

# Higher LR (faster, might be unstable)
LEARNING_RATE = 0.002
```

### 3. Architecture Modifications

You can modify the CNN architecture (still following constraints):

```python
# Add more filters
self.conv1 = nn.Conv2d(3, 96, kernel_size=3, padding=1)  # 64 → 96

# Add more layers (but watch for overfitting)
# Add Block 6, 7, etc.
```

### 4. Training Tricks

```python
# Longer patience for early stopping
PATIENCE = 20

# More epochs
NUM_EPOCHS = 150

# Different optimizer
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
```

---

## 📞 Getting Help

### Common Questions

**Q: How long will training take?**
A: With GPU: 30-90 minutes. Without GPU: Don't even try (100+ hours)

**Q: Can I stop and resume training?**
A: Not easily in Colab. Best to let it run completely.

**Q: What if my runtime disconnects?**
A: You'll need to restart training. Use Google Drive to save checkpoints.

**Q: Can I use Colab Pro?**
A: Yes! Colab Pro gives you better GPUs and longer runtime. Not required though.

**Q: Why is my accuracy low?**
A: Could be many reasons - check data loading, try different hyperparameters, train longer.

### Resources

- [Google Colab Documentation](https://colab.research.google.com/notebooks/intro.ipynb)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- Course lecture notes and PDF

---

## 🎯 Final Checklist

Before submission:

- [ ] Trained model with GPU enabled
- [ ] Validation accuracy > 50%
- [ ] Downloaded `model.pth` (40-50 MB)
- [ ] Tested `submission_model.ipynb` with model.pth
- [ ] Both files work together
- [ ] Created `GroupX_Assignment.zip`
- [ ] Added teammate names in comment after submission

**You're ready to submit! Good luck! 🚀**

---

## Quick Command Reference

```python
# Check GPU
import torch
print(torch.cuda.is_available())

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Download file
from google.colab import files
files.download('model.pth')

# Check file size
import os
print(f"Size: {os.path.getsize('model.pth') / (1024*1024):.2f} MB")

# List files
!ls -lh *.pkl *.pth

# Check memory usage
!nvidia-smi  # GPU memory
```
