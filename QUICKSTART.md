# Quick Start Guide

## Setup (5 minutes)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Place data files in project root
# - train-70_.pkl
# - validation-10_.pkl
```

## Training (2-3 hours with GPU)

```bash
# Single command to train
python train_model.py
```

This will:
- Load and augment training data
- Train the CNN model
- Use early stopping
- Save best model to `model.pth`
- Generate training plots

## Testing Your Submission

```bash
# Verify everything works
python test_submission.py
```

This ensures:
- Model loads correctly
- Predictions work
- Validation accuracy is computed
- Confusion matrix generated

## Submit

Create `GroupX_Assignment.zip` with:
1. `submission_model.py`
2. `model.pth`

---

## One-Liner Workflow

```bash
pip install -r requirements.txt && python train_model.py && python test_submission.py
```

## Expected Results

| Metric | Value |
|--------|-------|
| Training Time | 30-50 epochs |
| Validation Accuracy | 50-70% |
| Model Size | ~40-50 MB |

## Common Commands

```bash
# Explore data first (optional)
python explore_data.py

# Train model
python train_model.py

# Test submission
python test_submission.py

# Use model for prediction
python submission_model.py validation-10_.pkl
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Out of memory | Reduce `BATCH_SIZE` in `train_model.py` |
| Low accuracy | Train longer or adjust hyperparameters |
| Can't find data | Place `.pkl` files in project root |
| Model won't load | Run `test_submission.py` for diagnostics |

## Key Files

- `train_model.py` - Main training script
- `submission_model.py` - **SUBMIT THIS** + `model.pth`
- `test_submission.py` - Validate before submission
- `README.md` - Full documentation

---

**Need help?** See `README.md` for detailed documentation.
