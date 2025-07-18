import jax
import jax.numpy as jnp
from datasets import load_dataset, Audio
from typing import Dict, List, Iterator, Optional, Tuple, Union
import numpy as np
import json
from dataclasses import dataclass
import logging

from tokenizer.alpha.mask_utils import create_padding_mask, pad_sequences_left

logger = logging.getLogger(__name__)


@dataclass
class AudioBatch:
    """Container for a batch of audio data with metadata."""
    audio: jax.Array  # [B, T, 1] - padded audio waveforms
    audio_lengths: jax.Array  # [B] - original audio lengths
    mask: jax.Array  # [B, 1, 1, T] - attention mask
    text: List[str]  # [B] - text transcriptions
    language: List[str]  # [B] - language codes
    speaker_features: Optional[jax.Array] = None  # [B, D] - speaker embeddings (future)

    def to_dict(self) -> Dict[str, Union[jax.Array, List]]:
        """Convert batch to dictionary format."""
        data = {
            "audio": self.audio,
            "audio_lengths": self.audio_lengths,
            "mask": self.mask,
            "text": self.text,
            "language": self.language,
        }
        if self.speaker_features is not None:
            data["speaker_features"] = self.speaker_features
        return data


class EmiliaDataLoader:
    """Streaming data loader for Emilia dataset with batching and padding."""

    def __init__(
            self,
            batch_size: int = 8,
            sample_rate: int = 24000,
            max_duration: float = 30.0,  # Maximum audio duration in seconds
            num_workers: int = 4,
            shuffle_buffer_size: int = 10000,
            seed: int = 42,
            subset: Optional[str] = None,  # Optional subset identifier
            streaming: bool = True,
            cache_dir: Optional[str] = None,
    ):
        """Initialize the data loader.

        Args:
            batch_size: Number of samples per batch
            sample_rate: Target sample rate for audio (24000 or 48000)
            max_duration: Maximum audio duration to keep (in seconds)
            num_workers: Number of workers for data loading
            shuffle_buffer_size: Size of shuffle buffer for streaming
            seed: Random seed for shuffling
            subset: Optional subset of the dataset to load
            streaming: Whether to use streaming mode
            cache_dir: Optional cache directory for non-streaming mode
        """
        self.batch_size = batch_size
        self.sample_rate = sample_rate
        self.max_duration = max_duration
        self.max_samples = int(max_duration * sample_rate)
        self.num_workers = num_workers
        self.shuffle_buffer_size = shuffle_buffer_size
        self.seed = seed
        self.subset = subset
        self.streaming = streaming
        self.cache_dir = cache_dir

        # Load dataset
        self._load_dataset()

        # Initialize speaker feature extractor placeholder
        self.speaker_extractor = None  # To be implemented

    def _load_dataset(self):
        """Load the Emilia dataset with appropriate configuration."""
        logger.info(f"Loading Emilia dataset (streaming={self.streaming})")

        try:
            # Load the dataset
            self.dataset = load_dataset(
                "amphion/Emilia-Dataset",
                split="train",
                streaming=self.streaming,
                cache_dir=self.cache_dir,
            )

            # Apply subset filtering if specified
            if self.subset:
                self.dataset = self.dataset.filter(
                    lambda x: self._filter_subset(x)
                )

            # Shuffle if in streaming mode
            if self.streaming:
                self.dataset = self.dataset.shuffle(
                    seed=self.seed,
                    buffer_size=self.shuffle_buffer_size
                )

            # Apply preprocessing
            self.dataset = self.dataset.map(
                self._preprocess_sample,
                remove_columns=["__key__", "__url__"]            )

            # Filter by duration
            self.dataset = self.dataset.filter(
                lambda x: x["duration"] <= self.max_duration
            )

        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            raise

    def _filter_subset(self, sample: Dict) -> bool:
        """Filter samples based on subset criteria."""
        # Implement subset filtering logic here
        # For example, filter by language or other metadata
        return True

    def _preprocess_sample(self, sample: Dict) -> Dict:
        """Preprocess a single sample from the dataset."""
        # Parse JSON metadata
        metadata = json.loads(sample["json"]) if isinstance(sample["json"], str) else sample["json"]

        # Extract audio array
        audio_data = sample["mp3"]
        if isinstance(audio_data, dict) and "array" in audio_data:
            audio_array = audio_data["array"]
            original_sr = audio_data.get("sampling_rate", 48000)
        else:
            # Handle different audio formats
            audio_array = np.array(audio_data)
            original_sr = 48000  # Default assumption

        # Resample if necessary
        if original_sr != self.sample_rate:
            # Simple downsampling (for better quality, use librosa or torchaudio)
            resample_ratio = self.sample_rate / original_sr
            audio_array = audio_array[::int(1 / resample_ratio)] if resample_ratio < 1 else audio_array

        # Convert to float32 and normalize
        audio_array = audio_array.astype(np.float32)
        if audio_array.max() > 1.0:
            audio_array = audio_array / 32768.0  # Assuming 16-bit audio

        # Calculate duration
        duration = len(audio_array) / self.sample_rate

        return {
            "audio": audio_array,
            "text": metadata.get("text", ""),
            "language": metadata.get("language", "unknown"),
            "duration": duration,
            "original_metadata": metadata,  # Keep for potential future use
        }

    def _extract_speaker_features(self, audio_batch: np.ndarray) -> Optional[np.ndarray]:
        """Extract speaker features from audio batch.

        Args:
            audio_batch: Batch of audio arrays [B, T]

        Returns:
            Speaker features [B, D] or None if extractor not available
        """
        if self.speaker_extractor is None:
            return None

        # Placeholder for speaker feature extraction
        # This could use models like speaker encoders, x-vectors, etc.
        # For now, return None
        return None

    def _collate_batch(self, samples: List[Dict]) -> AudioBatch:
        """Collate samples into a batch with left-padding."""
        batch_size = len(samples)

        # Extract audio arrays and metadata
        audio_arrays = []
        texts = []
        languages = []

        for sample in samples:
            audio = sample["audio"]
            # Truncate if too long
            if len(audio) > self.max_samples:
                audio = audio[:self.max_samples]
            audio_arrays.append(audio)
            texts.append(sample["text"])
            languages.append(sample["language"])

        # Find max length in batch
        lengths = np.array([len(audio) for audio in audio_arrays])
        max_length = int(lengths.max())

        # Pad audio arrays (left-padding)
        padded_audio = []
        for audio, length in zip(audio_arrays, lengths):
            if length < max_length:
                pad_width = max_length - length
                padded = np.pad(audio, (pad_width, 0), mode='constant', constant_values=0)
            else:
                padded = audio
            # Add channel dimension
            padded = padded[:, np.newaxis]  # [T, 1]
            padded_audio.append(padded)

        # Stack into batch
        audio_batch = np.stack(padded_audio, axis=0)  # [B, T, 1]

        # Convert to JAX arrays
        audio_jax = jnp.array(audio_batch)
        lengths_jax = jnp.array(lengths)

        # Create attention mask (non-causal for encoder)
        mask = create_padding_mask(lengths_jax, max_length, causal=False)

        # Extract speaker features if available
        speaker_features = self._extract_speaker_features(audxio_batch[:, :, 0])
        if speaker_features is not None:
            speaker_features = jnp.array(speaker_features)

        return AudioBatch(
            audio=audio_jax,
            audio_lengths=lengths_jax,
            mask=mask,
            text=texts,
            language=languages,
            speaker_features=speaker_features,
        )

    def __iter__(self) -> Iterator[AudioBatch]:
        """Iterate over batches of data."""
        if self.streaming:
            # Streaming mode: collect samples into batches
            batch = []
            for sample in self.dataset:
                batch.append(sample)
                if len(batch) == self.batch_size:
                    yield self._collate_batch(batch)
                    batch = []

            # Yield remaining samples if any
            if batch:
                yield self._collate_batch(batch)
        else:
            # Non-streaming mode: use indices
            num_samples = len(self.dataset)
            indices = np.arange(num_samples)
            np.random.shuffle(indices)

            for i in range(0, num_samples, self.batch_size):
                batch_indices = indices[i:i + self.batch_size]
                batch = [self.dataset[int(idx)] for idx in batch_indices]
                yield self._collate_batch(batch)

    def __len__(self) -> Optional[int]:
        """Return dataset length if available (non-streaming mode)."""
        if not self.streaming:
            return len(self.dataset) // self.batch_size
        return None


# Example usage and testing
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)

    # Create data loader
    loader = EmiliaDataLoader(
        batch_size=4,
        sample_rate=24000,
        max_duration=10.0,
        streaming=True,
    )

    # Test iteration
    for i, batch in enumerate(loader):
        print(f"\nBatch {i}:")
        print(f"  Audio shape: {batch.audio.shape}")
        print(f"  Audio lengths: {batch.audio_lengths}")
        print(f"  Mask shape: {batch.mask.shape}")
        print(f"  Languages: {batch.language}")
        print(f"  Text samples: {[text[:50] + '...' if len(text) > 50 else text for text in batch.text]}")

        if i >= 2:  # Just show a few batches
            break