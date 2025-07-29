#!/usr/bin/env python3
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from tokenizer.utils.data.loader import create_emilia_ds, AudioConfig
import jax.numpy as jnp


def test_phoneme_loader():
    """Test the modified data loader with phoneme conversion."""
    print("Testing phoneme data loader...")

    # Create config with small batch for testing
    config = AudioConfig(
        dataset_name="amphion/Emilia-Dataset",
        split="train",
        sample_rate=24000,
        batch_size=4,  # Small batch for testing
        streaming=True,
    )

    # Create dataset
    dataset = create_emilia_ds(config)

    # Get first batch
    print("\nFetching first batch...")
    batch = next(iter(dataset))

    # Print batch keys
    print(f"\nBatch keys: {list(batch.keys())}")

    # Check audio data
    print(f"\nAudio shape: {batch['audio'].shape}")
    print(f"Audio dtype: {batch['audio'].dtype}")
    print(f"Audio lengths: {batch['lengths']}")

    # Check phoneme data
    print(f"\nPhoneme indices shape: {batch['phonemes'].shape}")
    print(f"Phoneme indices dtype: {batch['phonemes'].dtype}")
    print(f"Phoneme mask shape: {batch['phoneme_mask'].shape}")
    print(f"Phoneme lengths: {batch['phoneme_lengths']}")

    # Print first phoneme sequence as example
    print(f"\nFirst phoneme sequence (indices): {batch['phonemes'][0].tolist()}...")
    print(f"First phoneme mask: {batch['phoneme_mask'][0][:20].tolist()}...")

    # Verify JAX arrays
    assert isinstance(batch["audio"], jnp.ndarray), "Audio should be JAX array"
    assert isinstance(batch["phonemes"], jnp.ndarray), "Phonemes should be JAX array"
    assert isinstance(batch["phoneme_mask"], jnp.ndarray), (
        "Phoneme mask should be JAX array"
    )

    # Verify no text/language keys
    assert "text" not in batch, "Text key should not be in batch"
    assert "language" not in batch, "Language key should not be in batch"

    print("\n✓ All tests passed! Batch only contains phoneme data.")


if __name__ == "__main__":
    test_phoneme_loader()
