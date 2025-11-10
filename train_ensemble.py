"""
Train Multiple Models for Ensembling
This script trains multiple models with different random seeds
for ensemble predictions (highest accuracy method)

Expected gain: +2-5% over single model
"""

import os
import subprocess
import sys

# Configuration
NUM_MODELS = 3  # Train 3 models (can increase to 5 for better results)
RANDOM_SEEDS = [42, 123, 777, 999, 1337][:NUM_MODELS]
BASE_SCRIPT = 'train_model_improved.py'  # or 'train_model.py'

def train_single_model(seed, model_num):
    """Train a single model with given seed"""
    print("\n" + "="*70)
    print(f"Training Model {model_num}/{NUM_MODELS} with seed {seed}")
    print("="*70)

    # Modify the seed in the script or pass as argument
    # For simplicity, we'll create a temporary modified script

    with open(BASE_SCRIPT, 'r') as f:
        script_content = f.read()

    # Replace random seed
    modified_content = script_content.replace(
        f"RANDOM_SEED = 42",
        f"RANDOM_SEED = {seed}"
    )

    # Replace model save path
    modified_content = modified_content.replace(
        "'model_improved.pth'",
        f"'model_ensemble_{model_num}.pth'"
    )

    # Write temporary script
    temp_script = f'temp_train_{model_num}.py'
    with open(temp_script, 'w') as f:
        f.write(modified_content)

    # Run training
    result = subprocess.run([sys.executable, temp_script], capture_output=False)

    # Clean up
    os.remove(temp_script)

    if result.returncode == 0:
        print(f"\n✓ Model {model_num} trained successfully!")
        return True
    else:
        print(f"\n❌ Model {model_num} training failed!")
        return False


def main():
    print("="*70)
    print("ENSEMBLE TRAINING")
    print("="*70)
    print(f"Training {NUM_MODELS} models with different random seeds")
    print(f"Seeds: {RANDOM_SEEDS}")
    print(f"Base script: {BASE_SCRIPT}")
    print("\nThis will take approximately:")
    print(f"  {NUM_MODELS} x 1-2 hours = {NUM_MODELS*1}-{NUM_MODELS*2} hours total")
    print("\nYou can run them in parallel on different machines to save time!")
    print("="*70)

    response = input("\nProceed with training? (yes/no): ")
    if response.lower() not in ['yes', 'y']:
        print("Training cancelled.")
        return

    successful_models = []

    for i, seed in enumerate(RANDOM_SEEDS, 1):
        success = train_single_model(seed, i)
        if success:
            successful_models.append(i)

    print("\n" + "="*70)
    print("ENSEMBLE TRAINING COMPLETE")
    print("="*70)
    print(f"Successfully trained: {len(successful_models)}/{NUM_MODELS} models")
    print(f"Model files created:")
    for i in successful_models:
        print(f"  - model_ensemble_{i}.pth")

    print("\nNext steps:")
    print("1. Test each model individually on validation set")
    print("2. Use ensemble_predict.py to combine predictions")
    print("3. Submit the best performing model or ensemble")
    print("="*70)


if __name__ == '__main__':
    main()
