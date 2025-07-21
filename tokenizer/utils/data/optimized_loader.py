import sys
import os
# Add the TRC directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

import jax
import jax.numpy as jnp
from typing import Dict, List, Optional, Iterator, Any
from datasets import load_dataset, Audio
from dataclasses import dataclass
import json
import warnings

from tokenizer.alpha.mask_utils import pad_sequences_left, create_padding_mask, create_encoder_masks

# Suppress potential warnings
warnings.filterwarnings('ignore', category=UserWarning)


@dataclass
class AudioConfig:
    """Configuration for simple audio loading."""
    dataset_name: str = "hf-internal-testing/librispeech_asr_dummy"
    dataset_config: str = "clean"
    split: str = "validation"
    sample_rate: int = 24000
    batch_size: int = 8
    max_duration_seconds: float = 10.0
    streaming: bool = True


class OptimizedAudioLoader:
    """Simple audio loader using native dataset methods."""
    
    def __init__(self, config: AudioConfig):
        self.config = config
        self.max_samples = int(config.max_duration_seconds * config.sample_rate)
        
        # Load dataset with HuggingFace's native JAX support
        print(f"Loading dataset: {config.dataset_name}/{config.dataset_config}")
        self.dataset = load_dataset(
            config.dataset_name,
            config.dataset_config,
            split=config.split,
            streaming=config.streaming,
        )
        
        # Detect columns
        self._setup_audio_column()
        
        # Cast audio column to correct format
        if self.audio_column and hasattr(self.dataset, 'cast_column'):
            try:
                self.dataset = self.dataset.cast_column(
                    self.audio_column,
                    Audio(sampling_rate=config.sample_rate)
                )
                print(f"Cast audio column to {config.sample_rate}Hz")
            except Exception as e:
                print(f"Warning: Could not cast audio column: {e}")
    
    def _setup_audio_column(self):
        """Setup audio column based on dataset format."""
        # Get first sample to detect format
        if self.config.streaming:
            first_sample = next(iter(self.dataset))
        else:
            first_sample = self.dataset[0] if len(self.dataset) > 0 else {}
        
        # Detect columns
        self.audio_column = None
        self.metadata_column = None
        
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
        
        print(f"Detected audio column: {self.audio_column}")
        print(f"Detected metadata column: {self.metadata_column}")
    
    def _resample_jax(self, audio: jax.Array, orig_sr: int, target_sr: int) -> jax.Array:
        """Simple resampling using JAX operations."""
        if orig_sr == target_sr:
            return audio
        
        # Simple linear interpolation resampling
        ratio = target_sr / orig_sr
        old_length = audio.shape[0]
        new_length = int(old_length * ratio)
        
        # Create interpolation indices
        old_indices = jnp.arange(old_length)
        new_indices = jnp.linspace(0, old_length - 1, new_length)
        
        # Linear interpolation
        resampled = jnp.interp(new_indices, old_indices, audio)
        
        return resampled
    
    def _process_sample(self, sample: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process a single sample."""
        try:
            audio_array = None
            sr = self.config.sample_rate
            
            # Extract audio - HuggingFace already provides it as JAX array
            if self.audio_column and self.audio_column in sample:
                audio_data = sample[self.audio_column]
                
                if isinstance(audio_data, dict) and 'array' in audio_data:
                    # HuggingFace Audio feature format - convert to JAX
                    audio_array = jnp.array(audio_data['array'], dtype=jnp.float32)
                    sr = audio_data.get('sampling_rate', self.config.sample_rate)
                elif hasattr(audio_data, 'shape'):
                    # Already a JAX/numpy array
                    audio_array = jnp.array(audio_data, dtype=jnp.float32)
                else:
                    return None
                
                # Resample if needed
                if sr != self.config.sample_rate:
                    audio_array = self._resample_jax(audio_array, sr, self.config.sample_rate)
                
                # Convert to mono if needed
                if audio_array.ndim > 1:
                    audio_array = jnp.mean(audio_array, axis=0)
                
                # Truncate if too long
                if len(audio_array) > self.max_samples:
                    audio_array = audio_array[:self.max_samples]
                
                # Normalization
                max_val = jnp.abs(audio_array).max()
                if max_val > 0:
                    audio_array = audio_array * (0.95 / max_val)
            else:
                return None
            
            # Parse metadata
            text = None
            language = 'EN'
            
            if self.metadata_column and self.metadata_column in sample:
                try:
                    if self.metadata_column == 'json' and isinstance(sample[self.metadata_column], str):
                        metadata = json.loads(sample[self.metadata_column])
                        text = metadata.get('text', '')
                        language = metadata.get('language', metadata.get('locale', 'EN'))
                    elif isinstance(sample[self.metadata_column], dict):
                        metadata = sample[self.metadata_column]
                        text = metadata.get('text', '')
                        language = metadata.get('language', metadata.get('locale', 'EN'))
                    elif isinstance(sample[self.metadata_column], str):
                        text = sample[self.metadata_column]
                except:
                    pass
            
            result = {
                'audio': audio_array,
                'length': len(audio_array),
                'text': text,
                'language': language
            }
            
            return result
            
        except Exception as e:
            print(f"Sample processing error: {e}")
            return None
    
    def _create_batch(self, samples: List[Dict[str, Any]]) -> Dict[str, jax.Array]:
        """Create a padded batch from samples."""
        # Filter out None samples
        samples = [s for s in samples if s is not None]
        if not samples:
            return None
        
        # Get audio arrays and lengths
        audio_arrays = [s['audio'] for s in samples]
        lengths = [s['length'] for s in samples]
        
        # Find max length
        max_length = max(lengths)
        
        # Make divisible by 480 for 24kHz (50Hz output) or 960 for 48kHz
        downsample_factor = 480 if self.config.sample_rate == 24000 else 960
        if max_length % downsample_factor != 0:
            max_length = ((max_length // downsample_factor) + 1) * downsample_factor
        
        # Audio arrays are already JAX arrays
        audio_sequences = audio_arrays
        
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
        
        # Create masks
        mask = create_padding_mask(lengths_array, max_length, causal=False)
        
        # Create encoder-level masks
        encoder_mask, encoder_causal_mask = create_encoder_masks(
            lengths_array, max_length, downsample_factor
        )
        
        batch = {
            'audio': audio_batch,
            'lengths': lengths_array,
            'mask': mask,
            'encoder_mask': encoder_mask,
            'encoder_causal_mask': encoder_causal_mask,
        }
        
        # Add text and language metadata
        batch['meta_text'] = [s.get('text', '') for s in samples]
        batch['meta_language'] = [s.get('language', 'EN') for s in samples]
        
        return batch
    
    def __iter__(self) -> Iterator[Dict[str, jax.Array]]:
        """Iterate over batches."""
        batch_buffer = []
        
        for sample in self.dataset:
            processed = self._process_sample(sample)
            if processed is not None:
                batch_buffer.append(processed)
            
            # Yield batch when full
            if len(batch_buffer) >= self.config.batch_size:
                batch = self._create_batch(batch_buffer[:self.config.batch_size])
                if batch is not None:
                    yield batch
                batch_buffer = batch_buffer[self.config.batch_size:]
        
        # Yield remaining samples
        while batch_buffer:
            batch_size = min(len(batch_buffer), self.config.batch_size)
            batch = self._create_batch(batch_buffer[:batch_size])
            if batch is not None:
                yield batch
            batch_buffer = batch_buffer[batch_size:]


def create_optimized_emilia_loader(
    split: str = "train", 
    batch_size: int = 8,
    sample_rate: int = 24000,
    **kwargs
) -> OptimizedAudioLoader:
    """Create a simple loader for Emilia dataset."""
    config = AudioConfig(
        dataset_name="amphion/Emilia-Dataset",
        dataset_config="default",  # Emilia uses 'default' config
        split=split,
        sample_rate=sample_rate,
        batch_size=batch_size,
        streaming=True,  # Always use streaming for large datasets
        **kwargs
    )
    return OptimizedAudioLoader(config)


def test_optimized_loader():
    """Test the simple loader."""
    print("Testing simple Emilia loader...")
    
    loader = create_optimized_emilia_loader(
        batch_size=4,
        max_duration_seconds=6.0,
    )
    
    for i, batch in enumerate(loader):
        if i >= 5:  # Test 5 batches
            break
        
        print(f"\nBatch {i} shapes:")
        print(f"  Audio: {batch['audio'].shape}")
        print(f"  Lengths: {batch['lengths']}")
        print(f"  Has text metadata: {len(batch['meta_text'])} samples")
        print(f"  Languages: {batch['meta_language']}")
        
        # Print first text sample
        if batch['meta_text'][0]:
            print(f"  First text: {batch['meta_text'][0][:50]}...")
    
    print("\nTest completed successfully!")


if __name__ == "__main__":
    test_optimized_loader()