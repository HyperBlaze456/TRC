import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from datasets import load_dataset
from dotenv import load_dotenv
from huggingface_hub import login
import jax.numpy as jnp
from typing import Dict, List, Optional, Iterator, Any
import json

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
    """Simple loader that processes batches without padding."""
    
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
        
        # Load dataset exactly like pls_work.py
        print(f"Loading Emilia dataset (split: {split})...")
        self.dataset = load_dataset(
            "amphion/Emilia-Dataset",
            "default",
            split=split,
            streaming=True
        )
        
        # Enable multithreaded decoding for faster streaming
        import os
        num_threads = min(32, (os.cpu_count() or 1) + 4)
        print(f"Using {num_threads} threads for audio decoding")
        
        # Decode the dataset with multiple threads
        decoded_dataset = self.dataset.decode(num_threads=num_threads)
        
        # Use native HuggingFace batching
        self.batched_dataset = decoded_dataset.batch(batch_size=batch_size)
        print("Dataset loaded successfully!")
    
    def _extract_metadata(self, json_str: str) -> Dict[str, str]:
        """Extract text and locale from JSON metadata."""
        try:
            metadata = json.loads(json_str)
            return {
                'text': metadata.get('text', ''),
                'locale': metadata.get('language', metadata.get('locale', 'EN'))
            }
        except:
            return {'text': '', 'locale': 'EN'}
    
    def _process_batch(self, batch: Dict[str, List]) -> Dict[str, Any]:
        """Process a batch - extract audio arrays and metadata only."""
        # Debug: Print batch keys and types
        print(f"DEBUG: Batch keys: {list(batch.keys())}")
        
        # Extract audio arrays with debugging
        audio_arrays = []
        audio_data_list = batch.get('mp3', [])
        print(f"DEBUG: Number of mp3 entries: {len(audio_data_list)}")
        
        for i, audio_data in enumerate(audio_data_list):
            print(f"DEBUG: Audio {i} type: {type(audio_data)}")
            
            # Handle AudioDecoder objects
            if hasattr(audio_data, 'get_all_samples'):
                try:
                    # This is an AudioDecoder object
                    samples = audio_data.get_all_samples()
                    audio_array = jnp.array(samples.data, dtype=jnp.float32)
                    audio_arrays.append(audio_array)
                    print(f"DEBUG: Audio {i} decoded from AudioDecoder, shape: {audio_array.shape}, sr: {samples.sample_rate}")
                except Exception as e:
                    print(f"DEBUG: Audio {i} AudioDecoder decoding failed: {e}")
            
            # Handle dict format (older HuggingFace format)
            elif isinstance(audio_data, dict):
                print(f"DEBUG: Audio {i} keys: {list(audio_data.keys())}")
                if 'array' in audio_data:
                    # Convert to JAX array
                    audio_array = jnp.array(audio_data['array'], dtype=jnp.float32)
                    audio_arrays.append(audio_array)
                    print(f"DEBUG: Audio {i} shape: {audio_array.shape}")
                else:
                    print(f"DEBUG: Audio {i} missing 'array' key")
            
            # Try direct conversion if it's already an array
            else:
                try:
                    audio_array = jnp.array(audio_data, dtype=jnp.float32)
                    audio_arrays.append(audio_array)
                    print(f"DEBUG: Audio {i} converted directly, shape: {audio_array.shape}")
                except Exception as e:
                    print(f"DEBUG: Audio {i} conversion failed: {e}")
        
        # Extract metadata
        metadata_list = []
        for json_str in batch.get('json', []):
            metadata_list.append(self._extract_metadata(json_str))
        
        # Extract texts and locales
        texts = [m['text'] for m in metadata_list]
        locales = [m['locale'] for m in metadata_list]
        
        return {
            'audio': audio_arrays,  # List of JAX arrays (different lengths)
            'text': texts,
            'locale': locales,
            'batch_size': len(audio_arrays)
        }
    
    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Iterate over batches."""
        for batch in self.batched_dataset:
            yield self._process_batch(batch)


def test_simple_loader():
    """Test the simple loader."""
    print("Testing simple Emilia loader...")
    
    loader = SimpleEmiliaLoader(
        split="train",
        batch_size=4,
        sample_rate=24000,
    )
    
    # Test iteration
    for i, batch in enumerate(loader):
        print(f"\nBatch {i}:")
        print(f"  Number of audio samples: {batch['batch_size']}")
        print(f"  Audio shapes: {[a.shape for a in batch['audio']]}")
        print(f"  Number of texts: {len(batch['text'])}")
        print(f"  Locales: {batch['locale']}")
        
        # Show first text if available
        if batch['text'] and batch['text'][0]:
            print(f"  First text: {batch['text'][0][:50]}...")
        
        if i >= 2:  # Test a few batches
            break
    
    print("\nTest completed!")


def test_raw_batch():
    """Test raw batching like pls_work.py to debug."""
    print("Testing raw batching like pls_work.py...")
    
    ds = load_dataset(
        "amphion/Emilia-Dataset",
        "default",
        split="train",
        streaming=True
    )
    
    # Test native batching
    batched = ds.batch(batch_size=4)
    
    for i, batch in enumerate(batched):
        print(f"\nRaw Batch {i}:")
        print(f"  Type: {type(batch)}")
        print(f"  Keys: {list(batch.keys())}")
        
        # Check each key's content
        for key in list(batch.keys()):
            print(f"  {key}: type={type(batch[key])}, length={len(batch[key]) if isinstance(batch[key], list) else 'N/A'}")
            
            # Look at first item in detail
            if isinstance(batch[key], list) and len(batch[key]) > 0:
                first_item = batch[key][0]
                print(f"    First item type: {type(first_item)}")
                if isinstance(first_item, dict):
                    print(f"    First item keys: {list(first_item.keys())}")
                elif isinstance(first_item, str):
                    print(f"    First item preview: {first_item[:50]}...")
        
        if i >= 1:  # Just check first 2 batches
            break


if __name__ == "__main__":
    print("=== Testing raw batch first ===")
    test_raw_batch()
    
    print("\n\n=== Testing simple loader ===")
    test_simple_loader()