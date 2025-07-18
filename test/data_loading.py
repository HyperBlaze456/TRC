from datasets import load_dataset, Audio
import jax
import jax.numpy as jnp
from typing import Dict, Iterator, List, Tuple


class JAXAudioStreamingDataset:
    """Minimal wrapper around HuggingFace IterableDataset for audio streaming with JAX"""

    def __init__(
            self,
            dataset_name: str,
            split: str = "train",
            batch_size: int = 32,
            sampling_rate: int = 16000,
            drop_last_batch: bool = False,
            dtype: jnp.dtype = jnp.float32
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
        self.dtype = dtype

    def __iter__(self) -> Iterator[Dict[str, jax.Array]]:
        """Iterate over batched examples as JAX arrays"""
        # Create batched iterator
        batched_dataset = self.dataset.batch(
            batch_size=self.batch_size,
            drop_last_batch=self.drop_last_batch
        )

        for batch in batched_dataset:
            yield self._convert_to_jax(batch)

    def _convert_to_jax(self, batch: Dict[str, List]) -> Dict[str, jax.Array]:
        """Convert batch to JAX arrays"""
        jax_batch = {}

        # Convert audio arrays
        if "audio" in batch:
            audio_arrays = [jnp.array(audio["array"], dtype=self.dtype)
                            for audio in batch["audio"]]

            # Pad and stack
            padded_audio, lengths = self._pad_sequences(audio_arrays)
            jax_batch["audio"] = padded_audio
            jax_batch["audio_lengths"] = lengths

            # Keep paths if needed
            jax_batch["audio_paths"] = [audio["path"] for audio in batch["audio"]]

        # Convert other fields to JAX arrays where appropriate
        for key, value in batch.items():
            if key not in ["audio", "audio_paths"] and key not in jax_batch:
                try:
                    # Try to convert to JAX array
                    jax_batch[key] = jnp.array(value)
                except:
                    # Keep as is if conversion fails
                    jax_batch[key] = value

        return jax_batch

    def _pad_sequences(self, sequences: List[jax.Array]) -> Tuple[jax.Array, jax.Array]:
        """Pad sequences to same length and stack"""
        lengths = jnp.array([len(seq) for seq in sequences])
        max_length = jnp.max(lengths)

        # Pad each sequence
        padded = []
        for seq in sequences:
            pad_width = max_length - len(seq)
            if pad_width > 0:
                padded_seq = jnp.pad(seq, (0, pad_width), mode='constant')
            else:
                padded_seq = seq
            padded.append(padded_seq)

        # Stack into batch
        return jnp.stack(padded), lengths


# Minimal JAX example
def minimal_jax_example():
    """Simplest possible example of streaming audio data with JAX"""
    print("=== Minimal JAX Streaming Example ===")

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

        # Convert to JAX array
        audio_array = jnp.array(example["audio"]["array"], dtype=jnp.float32)
        sample_rate = example["audio"]["sampling_rate"]
        duration = len(audio_array) / sample_rate

        print(f"\nExample {i}:")
        print(f"  Audio shape: {audio_array.shape}")
        print(f"  Audio dtype: {audio_array.dtype}")
        print(f"  Sample rate: {sample_rate} Hz")
        print(f"  Duration: {duration:.2f} seconds")
        print(f"  Min/Max values: {jnp.min(audio_array):.4f} / {jnp.max(audio_array):.4f}")


# Batched JAX processing example
def batched_jax_example():
    """Example with JAX batch processing"""
    print("\n=== Batched JAX Processing Example ===")

    # Create streaming dataset with batching
    audio_dataset = JAXAudioStreamingDataset(
        dataset_name="PolyAI/minds14",
        split="train",
        batch_size=4,
        sampling_rate=16000,
        dtype=jnp.float32
    )

    # Process first 2 batches
    for batch_idx, batch in enumerate(audio_dataset):
        if batch_idx >= 2:
            break

        print(f"\nBatch {batch_idx}:")
        print(f"  Batch keys: {list(batch.keys())}")
        print(f"  Audio tensor shape: {batch['audio'].shape}")
        print(f"  Audio tensor dtype: {batch['audio'].dtype}")
        print(f"  Audio lengths: {batch['audio_lengths']}")

        # Simple JAX operations
        audio_mean = jnp.mean(batch['audio'], axis=1)
        audio_std = jnp.std(batch['audio'], axis=1)
        print(f"  Batch audio means shape: {audio_mean.shape}")
        print(f"  Batch audio stds shape: {audio_std.shape}")


# Advanced: Custom JAX preprocessing with .map()
def advanced_jax_preprocessing():
    """Example using .map() for on-the-fly JAX preprocessing"""
    print("\n=== Advanced JAX Preprocessing Example ===")

    dataset = load_dataset(
        "PolyAI/minds14",
        name="en-US",
        split="train",
        streaming=True
    )

    def extract_jax_features(examples):
        """Extract features using JAX operations"""
        features = []
        for audio in examples["audio"]:
            # Convert to JAX array
            arr = jnp.array(audio["array"], dtype=jnp.float32)

            # JAX-based feature extraction
            features.append({
                "mean": float(jnp.mean(arr)),
                "std": float(jnp.std(arr)),
                "max_amp": float(jnp.max(jnp.abs(arr))),
                "rms": float(jnp.sqrt(jnp.mean(arr ** 2))),
                "duration": len(arr) / audio["sampling_rate"]
            })
        examples["audio_features"] = features
        return examples

    # Apply preprocessing on-the-fly with batching
    processed_dataset = dataset.map(
        extract_jax_features,
        batched=True,
        batch_size=8
    )

    # Check first example
    first = next(iter(processed_dataset))
    print(f"Processed example keys: {list(first.keys())}")
    print(f"Audio features: {first['audio_features']}")


# Manual JAX batch creation with custom processing
def manual_jax_batch_iteration():
    """Manually handle JAX batching for maximum control"""
    print("\n=== Manual JAX Batch Iteration ===")

    dataset = load_dataset(
        "PolyAI/minds14",
        name="en-US",
        split="train",
        streaming=True
    )

    batch_size = 4
    batch = []

    @jax.jit
    def compute_batch_features(audio_batch):
        """JIT-compiled batch feature extraction"""
        # Compute spectral features
        fft = jnp.fft.rfft(audio_batch, axis=1)
        magnitude = jnp.abs(fft)

        # Simple features
        energy = jnp.sum(audio_batch ** 2, axis=1)
        spectral_centroid = jnp.sum(magnitude, axis=1) / (magnitude.shape[1] + 1e-8)

        return {
            "energy": energy,
            "spectral_centroid": spectral_centroid,
            "magnitude_mean": jnp.mean(magnitude, axis=1)
        }

    for idx, example in enumerate(dataset):
        batch.append(example)

        if len(batch) == batch_size:
            # Convert to JAX arrays and pad
            audio_arrays = [jnp.array(ex["audio"]["array"], dtype=jnp.float32)
                            for ex in batch]

            # Manual padding
            max_len = max(len(arr) for arr in audio_arrays)
            padded_arrays = []
            for arr in audio_arrays:
                if len(arr) < max_len:
                    padded = jnp.pad(arr, (0, max_len - len(arr)), mode='constant')
                else:
                    padded = arr
                padded_arrays.append(padded)

            # Stack into batch tensor
            audio_batch = jnp.stack(padded_arrays)

            # Compute features with JIT
            features = compute_batch_features(audio_batch)

            print(f"\nManual JAX batch {idx // batch_size}:")
            print(f"  Audio batch shape: {audio_batch.shape}")
            print(f"  Energy shape: {features['energy'].shape}")
            print(f"  Spectral centroid: {features['spectral_centroid']}")

            # Clear batch
            batch = []

        # Stop after 2 batches
        if idx >= 2 * batch_size:
            break


# Example: Preparing data for JAX/Flax model training
def jax_training_data_generator():
    """Generator for JAX/Flax training loops"""
    print("\n=== JAX Training Data Generator ===")

    dataset = JAXAudioStreamingDataset(
        dataset_name="PolyAI/minds14",
        split="train",
        batch_size=8,
        sampling_rate=16000,
        dtype=jnp.float32
    )

    def data_generator():
        """Infinite generator for training"""
        while True:
            for batch in dataset:
                # Create training batch with audio and labels
                audio = batch["audio"]

                # Convert labels if they exist
                if "intent_class" in batch:
                    labels = batch["intent_class"]
                else:
                    # Dummy labels for example
                    labels = jnp.zeros(len(audio), dtype=jnp.int32)

                # Return dict suitable for training
                yield {
                    "audio": audio,
                    "labels": labels,
                    "audio_lengths": batch["audio_lengths"]
                }

    # Example: get a few batches
    gen = data_generator()
    for i in range(3):
        batch = next(gen)
        print(f"\nTraining batch {i}:")
        print(f"  Audio shape: {batch['audio'].shape}")
        print(f"  Labels shape: {batch['labels'].shape}")
        print(f"  Lengths: {batch['audio_lengths']}")


if __name__ == "__main__":
    # Check JAX is working
    print(f"JAX version: {jax.__version__}")
    print(f"JAX devices: {jax.devices()}")

    # Run examples
    minimal_jax_example()
    batched_jax_example()
    advanced_jax_preprocessing()
    manual_jax_batch_iteration()
    jax_training_data_generator()