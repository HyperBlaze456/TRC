import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from datasets import load_dataset
from dotenv import load_dotenv
from huggingface_hub import login
import jax.numpy as jnp
from typing import Dict, List, Optional, Iterator, Any

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
    """Simple loader that starts like pls_work.py and defers all processing."""
    
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
        print("Dataset loaded successfully!")
        
        # We'll detect columns lazily during iteration
        self.audio_column = None
        self.metadata_column = None
        self._columns_detected = False
    
    def _detect_columns(self, sample: Dict[str, Any]):
        """Detect columns from first sample during iteration."""
        if self._columns_detected:
            return
            
        # Common column names
        audio_columns = ['audio', 'mp3', 'wav', 'flac', 'sound', 'recording']
        metadata_columns = ['json', 'metadata', 'text', 'transcription']
        
        for col in audio_columns:
            if col in sample:
                self.audio_column = col
                break
        
        for col in metadata_columns:
            if col in sample:
                self.metadata_column = col
                break
        
        self._columns_detected = True
        print(f"Detected columns - Audio: {self.audio_column}, Metadata: {self.metadata_column}")
    
    def _process_audio(self, audio_data: Any) -> Optional[jnp.ndarray]:
        """Process audio data lazily during iteration."""
        try:
            # Handle HuggingFace Audio feature format
            if isinstance(audio_data, dict) and 'array' in audio_data:
                audio_array = jnp.array(audio_data['array'], dtype=jnp.float32)
                sr = audio_data.get('sampling_rate', self.sample_rate)
                
                # Simple resampling if needed (we'll improve this later)
                if sr != self.sample_rate and sr > 0:
                    # For now, just truncate/pad - proper resampling can be added later
                    print(f"Warning: Audio at {sr}Hz, expected {self.sample_rate}Hz")
                
                # Convert to mono if needed
                if audio_array.ndim > 1:
                    audio_array = jnp.mean(audio_array, axis=0)
                
                # Truncate if too long
                if len(audio_array) > self.max_samples:
                    audio_array = audio_array[:self.max_samples]
                
                return audio_array
            else:
                return None
                
        except Exception as e:
            print(f"Audio processing error: {e}")
            return None
    
    def __iter__(self) -> Iterator[List[Dict[str, Any]]]:
        """Iterate over batches - simple version first."""
        batch = []
        
        for sample in self.dataset:
            # Detect columns from first sample
            if not self._columns_detected:
                self._detect_columns(sample)
            
            # For now, just yield raw batches like pls_work.py
            batch.append(sample)
            
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        
        # Yield remaining samples
        if batch:
            yield batch


def test_simple_loader():
    """Test the simple loader - should be as fast as pls_work.py"""
    print("Testing simple Emilia loader...")
    
    loader = SimpleEmiliaLoader(
        split="train",
        batch_size=16,
        sample_rate=24000,
    )
    
    # Test iteration - should start immediately!
    for i, batch in enumerate(loader):
        print(f"\nBatch {i}: {len(batch)} samples")
        
        # Show first sample structure
        if i == 0 and batch:
            print(f"Sample keys: {list(batch[0].keys())}")
            
            # Try processing first audio sample
            if loader.audio_column and loader.audio_column in batch[0]:
                audio = loader._process_audio(batch[0][loader.audio_column])
                if audio is not None:
                    print(f"Processed audio shape: {audio.shape}")
        
        if i >= 2:  # Test a few batches
            break
    
    print("\nTest completed!")


if __name__ == "__main__":
    test_simple_loader()