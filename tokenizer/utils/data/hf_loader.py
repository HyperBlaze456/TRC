"""Simple HuggingFace dataset loader for audio with JAX arrays and padding."""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, List, Optional, Iterator, Tuple, Any
import librosa
from datasets import load_dataset, Audio
from dataclasses import dataclass
import json
import io

from tokenizer.alpha.mask_utils import pad_sequences_left, create_padding_mask


@dataclass
class AudioConfig:
    """Simple configuration for audio loading."""
    dataset_name: str = "hf-internal-testing/librispeech_asr_dummy"
    dataset_config: str = "clean"
    split: str = "validation"
    sample_rate: int = 24000
    batch_size: int = 8
    max_duration_seconds: float = 10.0
    streaming: bool = True


class SimpleAudioLoader:
    """Simple audio dataset loader with batching and padding."""
    
    def __init__(self, config: AudioConfig):
        self.config = config
        self.max_samples = int(config.max_duration_seconds * config.sample_rate)
        
        # Load dataset
        self.dataset = load_dataset(
            config.dataset_name,
            config.dataset_config,
            split=config.split,
            streaming=config.streaming,
            trust_remote_code=True
        )
        
        # Detect dataset format and cast audio column accordingly
        self._setup_audio_column()
    
    def _setup_audio_column(self):
        """Setup audio column based on dataset format."""
        # Get first sample to detect format
        if self.config.streaming:
            first_sample = next(iter(self.dataset))
        else:
            first_sample = self.dataset[0] if len(self.dataset) > 0 else {}
        
        # Detect audio column name
        self.audio_column = None
        self.metadata_column = None
        
        # Common audio column names
        audio_columns = ['audio', 'mp3', 'wav', 'flac', 'sound', 'recording']
        metadata_columns = ['json', 'metadata', 'text', 'transcription']
        
        for col in audio_columns:
            if col in first_sample:
                self.audio_column = col
                break
        
        for col in metadata_columns:
            if col in first_sample:
                self.metadata_column = col
                break
        
        # Cast audio column if found
        if self.audio_column and hasattr(self.dataset, 'cast_column'):
            try:
                self.dataset = self.dataset.cast_column(
                    self.audio_column,
                    Audio(sampling_rate=self.config.sample_rate)
                )
            except Exception as e:
                print(f"Warning: Could not cast audio column: {e}")
    
    def process_sample(self, sample: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process a single sample from the dataset.
        
        Args:
            sample: Raw sample from dataset
            
        Returns:
            Processed sample with audio array or None if invalid
        """
        try:
            audio_array = None
            sr = None
            
            # Extract audio based on detected column
            if self.audio_column and self.audio_column in sample:
                audio_data = sample[self.audio_column]
                
                if isinstance(audio_data, dict) and 'array' in audio_data:
                    # Standard HuggingFace Audio format
                    audio_array = audio_data['array']
                    sr = audio_data.get('sampling_rate', self.config.sample_rate)
                elif isinstance(audio_data, (np.ndarray, list)):
                    # Direct array format
                    audio_array = np.array(audio_data)
                    sr = self.config.sample_rate
                elif isinstance(audio_data, bytes):
                    # Raw bytes (e.g., MP3 data)
                    audio_array, sr = librosa.load(io.BytesIO(audio_data), sr=None)
                else:
                    return None
                
                # Resample if needed
                if sr != self.config.sample_rate:
                    audio_array = librosa.resample(
                        audio_array,
                        orig_sr=sr,
                        target_sr=self.config.sample_rate,
                        res_type='kaiser_best'
                    )
            else:
                return None
            
            # Convert to mono if needed
            if audio_array.ndim > 1:
                audio_array = np.mean(audio_array, axis=0)
            
            # Check duration
            if len(audio_array) > self.max_samples:
                audio_array = audio_array[:self.max_samples]
            
            # Normalize
            max_val = np.abs(audio_array).max()
            if max_val > 0:
                audio_array = audio_array / max_val * 0.95
            
            # Parse metadata if available
            metadata = {}
            if self.metadata_column and self.metadata_column in sample:
                try:
                    if self.metadata_column == 'json' and isinstance(sample[self.metadata_column], str):
                        metadata = json.loads(sample[self.metadata_column])
                    elif isinstance(sample[self.metadata_column], dict):
                        metadata = sample[self.metadata_column]
                    elif isinstance(sample[self.metadata_column], str):
                        metadata = {'text': sample[self.metadata_column]}
                except Exception as e:
                    print(f"Warning: Could not parse metadata: {e}")
            
            result = {
                'audio': audio_array.astype(np.float32),
                'length': len(audio_array)
            }
            
            # Add metadata fields
            for key, value in metadata.items():
                result[f'meta_{key}'] = value
            
            return result
            
        except Exception as e:
            print(f"Error processing sample: {e}")
            return None
    
    def create_batch(self, samples: List[Dict[str, Any]]) -> Dict[str, jax.Array]:
        """Create a padded batch from samples.
        
        Args:
            samples: List of processed samples
            
        Returns:
            Batch dictionary with JAX arrays
        """
        # Get audio arrays and lengths
        audio_arrays = [s['audio'] for s in samples]
        lengths = [s['length'] for s in samples]
        
        # Find max length
        max_length = max(lengths)
        
        # Make divisible by 480 for 24kHz (50Hz output)
        if max_length % 480 != 0:
            max_length = ((max_length // 480) + 1) * 480
        
        # Convert to JAX arrays
        audio_sequences = [jnp.array(audio, dtype=jnp.float32) for audio in audio_arrays]
        
        # Pad sequences (left-padding)
        padded_audio, _ = pad_sequences_left(
            audio_sequences,
            max_length=max_length,
            pad_value=0.0
        )
        
        # Add channel dimension [B, T, 1]
        audio_batch = padded_audio[:, :, None]
        
        # Create lengths array
        lengths_array = jnp.array(lengths, dtype=jnp.int32)
        
        # Create mask
        mask = create_padding_mask(lengths_array, max_length, causal=False)
        
        batch = {
            'audio': audio_batch,
            'lengths': lengths_array,
            'mask': mask
        }
        
        # Collect metadata fields
        metadata_keys = set()
        for sample in samples:
            metadata_keys.update(k for k in sample.keys() if k.startswith('meta_'))
        
        # Add metadata to batch as lists
        for key in metadata_keys:
            batch[key] = [sample.get(key, None) for sample in samples]
        
        return batch
    
    def __iter__(self) -> Iterator[Dict[str, jax.Array]]:
        """Iterate over batches."""
        buffer = []
        
        for sample in self.dataset:
            processed = self.process_sample(sample)
            
            if processed is not None:
                buffer.append(processed)
            
            # Yield batch when full
            if len(buffer) >= self.config.batch_size:
                batch = self.create_batch(buffer[:self.config.batch_size])
                yield batch
                buffer = buffer[self.config.batch_size:]
        
        # Yield remaining if any
        if buffer:
            batch = self.create_batch(buffer)
            yield batch


# Helper functions for common datasets
def create_emilia_loader(
    language: str = "EN",
    split: str = "train",
    batch_size: int = 8,
    sample_rate: int = 24000,
    **kwargs
) -> SimpleAudioLoader:
    """Create a loader for Emilia dataset."""
    config = AudioConfig(
        dataset_name="amphion/Emilia-Dataset",
        dataset_config=language,
        split=split,
        sample_rate=sample_rate,
        batch_size=batch_size,
        **kwargs
    )
    return SimpleAudioLoader(config)


def create_librispeech_loader(
    config_name: str = "clean",
    split: str = "train.360",
    batch_size: int = 8,
    sample_rate: int = 24000,
    **kwargs
) -> SimpleAudioLoader:
    """Create a loader for LibriSpeech dataset."""
    config = AudioConfig(
        dataset_name="librispeech_asr",
        dataset_config=config_name,
        split=split,
        sample_rate=sample_rate,
        batch_size=batch_size,
        **kwargs
    )
    return SimpleAudioLoader(config)


def create_common_voice_loader(
    language: str = "en",
    split: str = "train",
    batch_size: int = 8,
    sample_rate: int = 24000,
    **kwargs
) -> SimpleAudioLoader:
    """Create a loader for Common Voice dataset."""
    config = AudioConfig(
        dataset_name="mozilla-foundation/common_voice_11_0",
        dataset_config=language,
        split=split,
        sample_rate=sample_rate,
        batch_size=batch_size,
        **kwargs
    )
    return SimpleAudioLoader(config)


# Test function
def test_simple_loader():
    """Test the simple audio loader with Emilia dataset."""
    # Test with Emilia dataset
    print("Testing with Emilia dataset...")
    
    # Create config directly since Emilia uses 'default' config
    config = AudioConfig(
        dataset_name="amphion/Emilia-Dataset",
        dataset_config="default",  # Emilia uses 'default' config
        split="train",
        batch_size=4,
        sample_rate=24000,
        max_duration_seconds=10.0,
        streaming=True  # Use streaming for large dataset
    )
    
    loader = SimpleAudioLoader(config)
    
    print(f"Detected audio column: {loader.audio_column}")
    print(f"Detected metadata column: {loader.metadata_column}")
    
    for i, batch in enumerate(loader):
        if i >= 2:  # Just test 2 batches
            break
        
        print(f"\nBatch {i}:")
        print(f"  Audio shape: {batch['audio'].shape}")
        print(f"  Lengths: {batch['lengths']}")
        print(f"  Mask shape: {batch['mask'].shape}")
        
        # Print metadata fields
        print("\n  Metadata fields:")
        for key in sorted(batch.keys()):
            if key.startswith('meta_'):
                field_name = key.replace('meta_', '')
                print(f"    {field_name}:")
                for j, value in enumerate(batch[key]):
                    if isinstance(value, str) and len(value) > 100:
                        print(f"      [{j}]: {value[:100]}...")
                    else:
                        print(f"      [{j}]: {value}")
        
        # Verify shapes
        B, T, C = batch['audio'].shape
        assert C == 1, f"Expected 1 channel, got {C}"
        assert batch['lengths'].shape == (B,), f"Invalid lengths shape"
        assert batch['mask'].shape == (B, 1, 1, T), f"Invalid mask shape"
        assert T % 480 == 0, f"Time dimension {T} not divisible by 480"
        
        # Check mask validity
        for b in range(B):
            mask_sum = int(batch['mask'][b, 0, 0, :].sum())
            actual_length = int(batch['lengths'][b])
            assert mask_sum == actual_length, f"Mask sum {mask_sum} != length {actual_length}"
        
        print("\n  ✓ Batch validation passed")
    
    print("\n✓ All tests passed!")


if __name__ == "__main__":
    test_simple_loader()