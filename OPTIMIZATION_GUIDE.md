# Performance Optimization Guide - Push Beyond 76%

Current Performance: **76.61% Validation Accuracy** ✅ (Very Good!)

Target: **80%+ Validation Accuracy** 🎯 (Top 3 Territory!)

## 🚀 Quick Wins (Easiest to Implement)

### 1. Improved Data Augmentation ⭐ (Expected gain: +1-3%)

**Current augmentation is basic. Add these:**

```python
# ENHANCED Data Augmentation
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(20),  # Increased from 15
    transforms.RandomAffine(
        degrees=0,
        translate=(0.15, 0.15),  # Increased from 0.1
        scale=(0.9, 1.1),  # NEW: Random scaling
        shear=10  # NEW: Random shear
    ),
    transforms.ColorJitter(
        brightness=0.3,  # Increased from 0.2
        contrast=0.3,
        saturation=0.3,
        hue=0.15  # Increased from 0.1
    ),
    transforms.RandomApply([
        transforms.GaussianBlur(kernel_size=3)  # NEW: Random blur
    ], p=0.3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```

### 2. Adjust Learning Rate & Optimizer ⭐⭐ (Expected gain: +1-2%)

**Try these combinations:**

**Option A: Lower Initial LR with Cosine Annealing**
```python
LEARNING_RATE = 0.0005  # Lower initial LR
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

# Replace ReduceLROnPlateau with CosineAnnealingWarmRestarts
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=10, T_mult=2, eta_min=1e-6
)
```

**Option B: SGD with Momentum (often better than Adam for final %)**
```python
LEARNING_RATE = 0.01
optimizer = optim.SGD(
    model.parameters(),
    lr=LEARNING_RATE,
    momentum=0.9,
    weight_decay=5e-4,
    nesterov=True
)

scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=NUM_EPOCHS, eta_min=1e-6
)
```

### 3. Fine-tune Dropout ⭐ (Expected gain: +0.5-1%)

Your current model uses `dropout_rate=0.5`. Try:

```python
# Slightly reduce dropout for better capacity
model = CustomCNN(num_classes=15, dropout_rate=0.4)  # Was 0.5
```

Or create variable dropout:
```python
self.dropout1 = nn.Dropout(0.3)  # Lower for early layers
self.dropout2 = nn.Dropout(0.4)
self.dropout3 = nn.Dropout(0.2)  # Lower before output
```

### 4. Train Longer with Better Early Stopping ⭐ (Expected gain: +1-2%)

```python
NUM_EPOCHS = 150  # Increase from 100
PATIENCE = 20  # Increase from 15
```

---

## 🏗️ Architecture Improvements (Medium Effort)

### 5. Deeper/Wider Network ⭐⭐⭐ (Expected gain: +2-4%)

**Add a 6th convolutional block:**

```python
# Add after Block 5 in CustomCNN
# Block 6: 2x2x512 -> 1x1x512
self.conv6 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
self.bn6 = nn.BatchNorm2d(512)
self.relu6 = nn.ReLU(inplace=True)
# No pooling - already at 2x2

# Update forward pass
x = self.relu6(self.bn6(self.conv6(x)))
```

**Or increase channel sizes:**

```python
# Instead of 64->128->256->512->512
# Try:     96->192->384->768->768
self.conv1 = nn.Conv2d(3, 96, kernel_size=3, padding=1)  # 64->96
self.conv2 = nn.Conv2d(96, 192, kernel_size=3, padding=1)  # 128->192
# etc...
```

### 6. Use Multiple Conv Layers per Block ⭐⭐ (Expected gain: +1-3%)

```python
# Block 1: Two conv layers before pooling
self.conv1a = nn.Conv2d(3, 64, kernel_size=3, padding=1)
self.bn1a = nn.BatchNorm2d(64)
self.conv1b = nn.Conv2d(64, 64, kernel_size=3, padding=1)
self.bn1b = nn.BatchNorm2d(64)
self.relu1 = nn.ReLU(inplace=True)
self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

# Forward:
x = self.relu1(self.bn1a(self.conv1a(x)))
x = self.relu1(self.bn1b(self.conv1b(x)))
x = self.pool1(x)
```

---

## 🎯 Advanced Techniques (More Effort, Big Gains)

### 7. Label Smoothing ⭐⭐⭐ (Expected gain: +0.5-1.5%)

```python
# Replace CrossEntropyLoss with Label Smoothing
class LabelSmoothingLoss(nn.Module):
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

# Use it
criterion = LabelSmoothingLoss(classes=15, smoothing=0.1)
```

### 8. Mixup Data Augmentation ⭐⭐⭐⭐ (Expected gain: +2-4%)

```python
def mixup_data(x, y, alpha=0.2):
    """Mixup augmentation"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# In training loop:
images, labels = images.to(device), labels.to(device)
images, targets_a, targets_b, lam = mixup_data(images, labels, alpha=0.2)
outputs = model(images)
loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
```

### 9. Test-Time Augmentation (TTA) ⭐⭐⭐ (Expected gain: +1-2%)

```python
def predict_with_tta(model, images, device, n_augments=5):
    """Make predictions with test-time augmentation"""
    model.eval()
    predictions = []

    # Original prediction
    with torch.no_grad():
        outputs = model(images)
        predictions.append(torch.softmax(outputs, dim=1))

    # Augmented predictions
    for _ in range(n_augments - 1):
        # Random horizontal flip
        if random.random() > 0.5:
            aug_images = torch.flip(images, dims=[3])
        else:
            aug_images = images

        with torch.no_grad():
            outputs = model(aug_images)
            predictions.append(torch.softmax(outputs, dim=1))

    # Average predictions
    final_pred = torch.stack(predictions).mean(dim=0)
    return final_pred.argmax(dim=1)
```

### 10. Multiple Models Ensemble ⭐⭐⭐⭐⭐ (Expected gain: +2-5%)

**Train 3-5 models with different seeds:**

```python
# Train model 1
RANDOM_SEED = 42
# ... train and save as model1.pth

# Train model 2
RANDOM_SEED = 123
# ... train and save as model2.pth

# Train model 3
RANDOM_SEED = 777
# ... train and save as model3.pth

# Ensemble prediction
def ensemble_predict(models, test_data):
    all_predictions = []

    for model in models:
        preds = predict(model, test_data)
        all_predictions.append(preds)

    # Majority voting
    all_predictions = np.array(all_predictions)
    final_predictions = []

    for i in range(all_predictions.shape[1]):
        votes = all_predictions[:, i]
        final_predictions.append(np.bincount(votes).argmax())

    return np.array(final_predictions)

# Load and ensemble
model1 = load_model('model1.pth')
model2 = load_model('model2.pth')
model3 = load_model('model3.pth')

predictions = ensemble_predict([model1, model2, model3], test_data)
```

---

## 📊 Recommended Combination Strategy

### **Strategy A: Quick Improvements (1-2 hours)**
Implement these for immediate gains to ~78-79%:

1. ✅ Enhanced data augmentation (see #1)
2. ✅ Lower learning rate (0.0005) with AdamW
3. ✅ Reduce dropout to 0.4
4. ✅ Train for 150 epochs with patience=20

**Expected result: 78-79% (+1.5-2.5%)**

### **Strategy B: Best Single Model (3-5 hours)**
For maximum single model performance ~80-81%:

1. ✅ All from Strategy A
2. ✅ Add 6th conv block OR widen network
3. ✅ Implement Label Smoothing
4. ✅ Implement Mixup augmentation
5. ✅ Use SGD with Nesterov momentum

**Expected result: 80-81% (+3.5-4.5%)**

### **Strategy C: Compete for #1 (6-10 hours)**
Maximum possible performance ~82-84%:

1. ✅ All from Strategy B
2. ✅ Train 5 different models with different seeds
3. ✅ Ensemble all 5 models
4. ✅ Add TTA during final prediction
5. ✅ Use both horizontal flips AND slight rotations in TTA

**Expected result: 82-84% (+5.5-7.5%)**

---

## 🔧 Specific Hyperparameters to Try

### Configuration 1: Conservative (Reliable ~78-79%)
```python
BATCH_SIZE = 64
LEARNING_RATE = 0.0005
WEIGHT_DECAY = 1e-4
DROPOUT_RATE = 0.4
NUM_EPOCHS = 150
PATIENCE = 20
optimizer = optim.AdamW
```

### Configuration 2: Aggressive (Risk/Reward ~79-81%)
```python
BATCH_SIZE = 96
LEARNING_RATE = 0.01
WEIGHT_DECAY = 5e-4
DROPOUT_RATE = 0.3
NUM_EPOCHS = 150
PATIENCE = 25
optimizer = optim.SGD with momentum=0.9, nesterov=True
```

### Configuration 3: Hybrid (Balanced ~80-82%)
```python
BATCH_SIZE = 80
LEARNING_RATE = 0.001 -> decay to 0.0001
WEIGHT_DECAY = 3e-4
DROPOUT_RATE = 0.35
NUM_EPOCHS = 200
PATIENCE = 30
optimizer = Start with SGD, switch to Adam at epoch 100
```

---

## 📈 Training Tips

### 1. Monitor Overfitting
```python
# Good: Small gap
Train Acc: 0.82, Val Acc: 0.80  ✅ Gap = 2%

# Bad: Large gap
Train Acc: 0.95, Val Acc: 0.76  ❌ Gap = 19% (overfitting!)
```

If overfitting:
- Increase dropout
- Add more data augmentation
- Reduce model capacity
- Increase weight decay

### 2. Learning Rate Finder
```python
# Try different learning rates and plot loss
lrs = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05]
# The best LR is usually where loss decreases fastest
```

### 3. Save Multiple Checkpoints
```python
# Don't just save best validation loss
# Also save:
# - Best validation accuracy
# - Last 5 epochs (for ensembling)
# - Every 10 epochs (for analysis)
```

---

## ⚡ Quick Implementation Plan

**Day 1 (2 hours):**
- Implement enhanced data augmentation
- Adjust learning rate to 0.0005
- Reduce dropout to 0.4
- Start training (150 epochs)

**Day 2 (3 hours):**
- Implement label smoothing
- Implement mixup
- Retrain with new techniques
- Compare with previous best

**Day 3 (4 hours):**
- Train 3 models with different seeds
- Implement ensemble prediction
- Test on validation set
- Select best approach

**Day 4 (1 hour):**
- Final training run
- Prepare submission
- Double-check everything works

---

## 🎯 My Recommendation for You

Since you already have **76.61%**, here's what I'd do:

**Immediate (Tonight - 2 hours):**
1. Copy your current training code
2. Change only these 3 things:
   ```python
   LEARNING_RATE = 0.0005  # Was 0.001
   DROPOUT_RATE = 0.4  # Was 0.5
   NUM_EPOCHS = 150  # Was 100
   ```
3. Add enhanced data augmentation (copy from #1 above)
4. Start training

**Expected result: ~78-79%**

**If you have more time (Tomorrow - 3 hours):**
5. Implement mixup augmentation (#8)
6. Implement label smoothing (#7)
7. Retrain

**Expected result: ~80-81%**

**For Top 3 (Weekend - 8 hours):**
8. Train 3-5 models with different random seeds
9. Ensemble them
10. Add TTA

**Expected result: ~82-84%**

---

## 📝 Testing Your Changes

After each improvement, test on validation set:

```python
# Current baseline
Val Acc: 0.7661

# After data aug + LR + dropout
Val Acc: 0.78XX  # Target

# After mixup + label smoothing
Val Acc: 0.80XX  # Target

# After ensemble
Val Acc: 0.82XX  # Top 3!
```

Want me to create an optimized training notebook with these improvements ready to run?
