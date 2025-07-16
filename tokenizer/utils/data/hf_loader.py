from datasets import load_dataset, Audio
import numpy as np
from typing import Dict, Iterator, List


class AudioStreamingDataset:
    """Minimal wrapper around HuggingFace IterableDataset for audio streaming"""

    def __init__(
            self,
            dataset_name: str,
            split: str = "train",
            batch_size: int = 32,
            sampling_rate: int = 16000,
            drop_last_batch: bool = False
    ):
        # Load dataset in streaming mode - creates an IterableDataset
        self.dataset = load_dataset(
            dataset_name,
            split=split,
            streaming=True
        )

        # Cast audio column to desired sampling rate (resampling happens on-the-fly)
        if "audio" in self.dataset.features:
            self.dataset = self.dataset.cast_column(
                "audio",
                Audio(sampling_rate=sampling_rate)
            )

        self.batch_size = batch_size
        self.drop_last_batch = drop_last_batch
        self.sampling_rate = sampling_rate

    def __iter__(self) -> Iterator[Dict[str, List]]:
        """Iterate over batched examples"""
        # Create batched iterator
        batched_dataset = self.dataset.batch(
            batch_size=self.batch_size,
            drop_last_batch=self.drop_last_batch
        )

        for batch in batched_dataset:
            yield batch

    def preprocess_batch(self, batch: Dict[str, List]) -> Dict[str, np.ndarray]:
        """Basic preprocessing - extract audio arrays and stack them"""
        if "audio" in batch:
            # Extract audio arrays from the batch
            # Each audio item has 'array', 'path', and 'sampling_rate' keys
            audio_arrays = [audio["array"] for audio in batch["audio"]]

            # Find max length for padding
            max_length = max(len(arr) for arr in audio_arrays)

            # Pad all arrays to same length
            padded_arrays = []
            for arr in audio_arrays:
                if len(arr) < max_length:
                    padded = np.pad(arr, (0, max_length - len(arr)), mode='constant')
                else:
                    padded = arr
                padded_arrays.append(padded)

            # Stack into batch tensor
            batch["audio_tensor"] = np.stack(padded_arrays)

            # Store original lengths for masking
            batch["audio_lengths"] = np.array([len(arr) for arr in audio_arrays])

        return batch


# Minimal usage example
def minimal_example():
    """Simplest possible example of streaming audio data"""
    print("=== Minimal Streaming Example ===")

    # Load a small audio dataset
    dataset = load_dataset(
        "PolyAI/minds14",  # Multi-lingual banking intent dataset
        name="en-US",  # English subset
        split="train",
        streaming=True
    )

    # Iterate over first 3 examples
    for i, example in enumerate(dataset):
        if i >= 3:
            break

        audio_array = example["audio"]["array"]
        sample_rate = example["audio"]["sampling_rate"]
        duration = len(audio_array) / sample_rate

        print(f"\nExample {i}:")
        print(f"  Audio shape: {audio_array.shape}")
        print(f"  Sample rate: {sample_rate} Hz")
        print(f"  Duration: {duration:.2f} seconds")
        print(f"  Intent: {example.get('intent_class', 'N/A')}")


# Batched processing example
def batched_example():
    """Example with batch processing"""
    print("\n=== Batched Processing Example ===")

    # Create streaming dataset with batching
    audio_dataset = AudioStreamingDataset(
        dataset_name="PolyAI/minds14",
        split="train",
        batch_size=4,
        sampling_rate=16000
    )

    # Process first 2 batches
    for batch_idx, batch in enumerate(audio_dataset):
        if batch_idx >= 2:
            break

        # Apply preprocessing
        processed = audio_dataset.preprocess_batch(batch)

        print(f"\nBatch {batch_idx}:")
        print(f"  Batch keys: {list(batch.keys())}")
        print(f"  Audio tensor shape: {processed['audio_tensor'].shape}")
        print(f"  Audio lengths: {processed['audio_lengths']}")


# Advanced: Custom preprocessing with .map()
def advanced_preprocessing_example():
    """Example using .map() for on-the-fly preprocessing"""
    print("\n=== Advanced Preprocessing Example ===")

    dataset = load_dataset(
        "PolyAI/minds14",
        name="en-US",
        split="train",
        streaming=True
    )

    def extract_features(examples):
        """Extract simple features from audio"""
        # This runs on batches when batched=True
        features = []
        for audio in examples["audio"]:
            arr = audio["array"]
            # Simple features: mean, std, max amplitude
            features.append({
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "max_amp": float(np.max(np.abs(arr))),
                "duration": len(arr) / audio["sampling_rate"]
            })
        examples["audio_features"] = features
        return examples

    # Apply preprocessing on-the-fly with batching
    processed_dataset = dataset.map(
        extract_features,
        batched=True,
        batch_size=8
    )

    # Check first example
    first = next(iter(processed_dataset))
    print(f"Processed example keys: {list(first.keys())}")
    print(f"Audio features: {first['audio_features']}")


# Manual iteration with custom batch handling
def manual_batch_iteration():
    """Manually handle batching for maximum control"""
    print("\n=== Manual Batch Iteration ===")

    dataset = load_dataset(
        "PolyAI/minds14",
        name="en-US",
        split="train",
        streaming=True
    )

    batch_size = 4
    batch = []

    for idx, example in enumerate(dataset):
        batch.append(example)

        if len(batch) == batch_size:
            # Process batch
            audio_arrays = [ex["audio"]["array"] for ex in batch]
            labels = [ex["intent_class"] for ex in batch]

            print(f"\nManual batch {idx // batch_size}:")
            print(f"  Audio shapes: {[arr.shape for arr in audio_arrays]}")
            print(f"  Labels: {labels}")

            # Clear batch
            batch = []

        # Stop after 2 batches
        if idx >= 2 * batch_size:
            break


if __name__ == "__main__":
    # Run examples
    minimal_example()
    batched_example()
    advanced_preprocessing_example()
    manual_batch_iteration()