"""Test training with reduced batch size to isolate memory issues."""

import os
import sys

# Set JAX traceback filtering off for detailed error traces
os.environ["JAX_TRACEBACK_FILTERING"] = "off"

# Add the TRC directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from train import TrainingConfig, train

if __name__ == "__main__":
    # Test with progressively smaller batch sizes
    batch_sizes = [8, 4, 2, 1]

    for batch_size in batch_sizes:
        print(f"\n{'=' * 60}")
        print(f"Testing with batch size: {batch_size}")
        print(f"{'=' * 60}\n")

        config = TrainingConfig(
            batch_size=batch_size,
            use_wandb=False,  # Disable W&B for testing
            profile_first_n_steps=1,  # Just profile first step
            log_every=1,  # Log every step
        )

        try:
            train(config)
        except Exception as e:
            print(f"Failed with batch size {batch_size}")
            print(f"Error: {e}")
            continue
