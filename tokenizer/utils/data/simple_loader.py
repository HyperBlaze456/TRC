import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from datasets import load_dataset
from dotenv import load_dotenv
from huggingface_hub import login
import jax.numpy as jnp
from typing import Dict, List, Optional, Iterator, Any
import json

from tokenizer.alpha.mask_utils import pad_sequences_left, create_padding_mask, create_encoder_masks

# Load environment variables
load_dotenv()
hf_token = os.getenv("HF_TOKEN")

# Set token globally for HuggingFace
if hf_token:
    print(f"Setting HF token (first 8 chars): {hf_token[:8]}...")
    try:
        login(token=hf_token)
        os.environ["HF_TOKEN"] = hf_token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token
        print("HF token set successfully")
    except Exception as e:
        print(f"Warning: Failed to login with HF token: {e}")


class SimpleEmiliaLoader:
    """Simple loader that uses native batching and adds preprocessing."""
    
    def __init__(
        self,
        split: str = "train",
        batch_size: int = 16,
        sample_rate: int = 24000,
        max_duration_seconds: float = 10.0,
    ):
        self.batch_size = batch_size
        self.sample_rate = sample_rate
        self.max_samples = int(max_duration_seconds * sample_rate)
        
        # Load dataset exactly like pls_work.py - no preprocessing!
        print(f"Loading Emilia dataset (split: {split})...")
        self.dataset = load_dataset(
            "amphion/Emilia-Dataset",
            "default",
            split=split,
            streaming=True
        )
        
        # Use native HuggingFace batching
        self.batched_dataset = self.dataset.batch(batch_size=batch_size)
        print("Dataset loaded successfully!")
        
        # For Emilia dataset, we know the columns
        self.audio_column = 'mp3'  # Emilia uses mp3 column
        self.metadata_column = 'json'  # Emilia uses json column
    
    def _process_audio_batch(self, audio_list: List[Any]) -> Optional[List[jnp.ndarray]]:
        """Process a batch of audio data."""
        processed_audios = []
        
        for audio_data in audio_list:
            try:
                # Handle HuggingFace Audio feature format
                if isinstance(audio_data, dict) and 'array' in audio_data:
                    audio_array = jnp.array(audio_data['array'], dtype=jnp.float32)
                    sr = audio_data.get('sampling_rate', self.sample_rate)
                    
                    # TODO: Add proper resampling if needed
                    if sr != self.sample_rate and sr > 0:
                        # For now, skip samples with wrong sample rate
                        continue
                    
                    # Convert to mono if needed
                    if audio_array.ndim > 1:
                        audio_array = jnp.mean(audio_array, axis=0)
                    
                    # Truncate if too long
                    if len(audio_array) > self.max_samples:
                        audio_array = audio_array[:self.max_samples]
                    
                    # Normalize
                    max_val = jnp.abs(audio_array).max()
                    if max_val > 0:
                        audio_array = audio_array * (0.95 / max_val)
                    
                    processed_audios.append(audio_array)
                    
            except Exception as e:
                print(f"Audio processing error: {e}")
                continue
        
        return processed_audios if processed_audios else None
    
    def _parse_metadata_batch(self, metadata_list: List[Any]) -> tuple:
        """Parse metadata from batch."""
        texts = []
        languages = []
        
        for metadata in metadata_list:
            text = None
            language = 'EN'
            
            try:
                if isinstance(metadata, str):
                    metadata_dict = json.loads(metadata)
                    text = metadata_dict.get('text', '')
                    language = metadata_dict.get('language', metadata_dict.get('locale', 'EN'))
                elif isinstance(metadata, dict):
                    text = metadata.get('text', '')
                    language = metadata.get('language', metadata.get('locale', 'EN'))
            except:
                pass
            
            texts.append(text or '')
            languages.append(language)
        
        return texts, languages
    
    def _create_batch_dict(self, batch: Dict[str, List]) -> Optional[Dict[str, Any]]:
        """Create a properly formatted batch dictionary with padding and masks."""
        # Process audio
        if self.audio_column not in batch:
            return None
            
        audio_list = self._process_audio_batch(batch[self.audio_column])
        if not audio_list:
            return None
        
        # Get lengths
        lengths = [len(audio) for audio in audio_list]
        max_length = max(lengths)
        
        # Make divisible by downsample factor
        downsample_factor = 480 if self.sample_rate == 24000 else 960
        if max_length % downsample_factor != 0:
            max_length = ((max_length // downsample_factor) + 1) * downsample_factor
        
        # Pad sequences (left-padding)
        padded_audio, _ = pad_sequences_left(
            audio_list,
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
        
        # Parse metadata
        texts, languages = [], []
        if self.metadata_column in batch:
            texts, languages = self._parse_metadata_batch(batch[self.metadata_column])
        
        return {
            'audio': audio_batch,
            'lengths': lengths_array,
            'mask': mask,
            'encoder_mask': encoder_mask,
            'encoder_causal_mask': encoder_causal_mask,
            'meta_text': texts,
            'meta_language': languages,
        }
    
    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Iterate over processed batches."""
        for batch in self.batched_dataset:
            # batch is a dict with lists as values
            processed_batch = self._create_batch_dict(batch)
            if processed_batch is not None:
                yield processed_batch


def create_emilia_loader(
    split: str = "train",
    batch_size: int = 8,
    sample_rate: int = 24000,
    max_duration_seconds: float = 10.0,
) -> SimpleEmiliaLoader:
    """Create a simple Emilia dataset loader."""
    return SimpleEmiliaLoader(
        split=split,
        batch_size=batch_size,
        sample_rate=sample_rate,
        max_duration_seconds=max_duration_seconds,
    )


def test_simple_loader():
    """Test the simple loader with preprocessing."""
    print("Testing simple Emilia loader with preprocessing...")
    
    loader = SimpleEmiliaLoader(
        split="train",
        batch_size=8,
        sample_rate=24000,
        max_duration_seconds=6.0,
    )
    
    # Test iteration - should start immediately!
    for i, batch in enumerate(loader):
        print(f"\nBatch {i}:")
        print(f"  Audio shape: {batch['audio'].shape}")
        print(f"  Lengths: {batch['lengths']}")
        print(f"  Mask shape: {batch['mask'].shape}")
        print(f"  Encoder mask shape: {batch['encoder_mask'].shape}")
        print(f"  Encoder causal mask shape: {batch['encoder_causal_mask'].shape}")
        print(f"  Number of texts: {len(batch['meta_text'])}")
        print(f"  Languages: {batch['meta_language']}")
        
        # Show first text if available
        if batch['meta_text'] and batch['meta_text'][0]:
            print(f"  First text: {batch['meta_text'][0][:50]}...")
        
        if i >= 2:  # Test a few batches
            break
    
    print("\nTest completed!")


if __name__ == "__main__":
    test_simple_loader()