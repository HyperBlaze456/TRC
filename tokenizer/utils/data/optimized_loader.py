"""Optimized audio data loader with streaming, parallel processing, and profiling."""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, List, Optional, Iterator, Tuple, Any, Callable
import librosa
from datasets import load_dataset, Audio
from dataclasses import dataclass
import json
import io
import time
from concurrent.futures import ThreadPoolExecutor, Future
from queue import Queue
import threading
from collections import deque
import warnings

from tokenizer.alpha.mask_utils import pad_sequences_left, create_padding_mask, create_encoder_masks

# Suppress librosa warnings for performance
warnings.filterwarnings('ignore', category=UserWarning, module='librosa')


@dataclass
class AudioConfig:
    """Configuration for optimized audio loading."""
    dataset_name: str = "hf-internal-testing/librispeech_asr_dummy"
    dataset_config: str = "clean"
    split: str = "validation"
    sample_rate: int = 24000
    batch_size: int = 8
    max_duration_seconds: float = 10.0
    streaming: bool = True
    # Optimization parameters
    num_workers: int = 4  # Number of parallel audio decoders
    prefetch_batches: int = 2  # Number of batches to prefetch
    decode_timeout: float = 30.0  # Timeout for audio decoding
    # Profiling parameters
    profile: bool = True
    profile_interval: int = 10  # Print stats every N batches


class ProfilingStats:
    """Track performance metrics."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.fetch_times = []
        self.decode_times = []
        self.batch_times = []
        self.total_samples = 0
        self.failed_samples = 0
        self.start_time = time.time()
    
    def print_stats(self, batch_idx: int):
        """Print current statistics."""
        elapsed = time.time() - self.start_time
        
        if self.fetch_times:
            avg_fetch = np.mean(self.fetch_times[-100:])  # Last 100 samples
        else:
            avg_fetch = 0
            
        if self.decode_times:
            avg_decode = np.mean(self.decode_times[-100:])
        else:
            avg_decode = 0
            
        if self.batch_times:
            avg_batch = np.mean(self.batch_times[-10:])  # Last 10 batches
        else:
            avg_batch = 0
        
        samples_per_sec = self.total_samples / elapsed if elapsed > 0 else 0
        
        print(f"\n=== Profiling Stats (Batch {batch_idx}) ===")
        print(f"Total time: {elapsed:.1f}s")
        print(f"Total samples: {self.total_samples} ({self.failed_samples} failed)")
        print(f"Throughput: {samples_per_sec:.1f} samples/sec")
        print(f"Avg fetch time: {avg_fetch*1000:.1f}ms")
        print(f"Avg decode time: {avg_decode*1000:.1f}ms") 
        print(f"Avg batch creation: {avg_batch*1000:.1f}ms")
        print(f"Pipeline efficiency: {(avg_decode / (avg_fetch + avg_decode + 1e-6)) * 100:.1f}%")
        print("=" * 40)


class OptimizedAudioLoader:
    """Optimized audio loader with parallel processing and profiling."""
    
    def __init__(self, config: AudioConfig):
        self.config = config
        self.max_samples = int(config.max_duration_seconds * config.sample_rate)
        self.stats = ProfilingStats() if config.profile else None
        
        # Load dataset
        print(f"Loading dataset: {config.dataset_name}/{config.dataset_config}")
        self.dataset = load_dataset(
            config.dataset_name,
            config.dataset_config,
            split=config.split,
            streaming=config.streaming,
        )
        
        # Detect columns
        self._setup_audio_column()
        
        # Setup parallel processing
        self.executor = ThreadPoolExecutor(max_workers=config.num_workers)
        self.decode_queue = Queue(maxsize=config.prefetch_batches * config.batch_size * 2)
        self.batch_queue = Queue(maxsize=config.prefetch_batches)
        
        # Start background threads
        self._start_workers()
    
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
        
        # Cast audio column if streaming not enabled
        if self.audio_column and not self.config.streaming and hasattr(self.dataset, 'cast_column'):
            try:
                self.dataset = self.dataset.cast_column(
                    self.audio_column,
                    Audio(sampling_rate=self.config.sample_rate)
                )
            except Exception as e:
                print(f"Warning: Could not cast audio column: {e}")
    
    def _start_workers(self):
        """Start background worker threads."""
        # Start dataset fetcher thread
        self.fetch_thread = threading.Thread(target=self._fetch_worker, daemon=True)
        self.fetch_thread.start()
        
        # Start batch creator thread  
        self.batch_thread = threading.Thread(target=self._batch_worker, daemon=True)
        self.batch_thread.start()
    
    def _fetch_worker(self):
        """Background thread that fetches samples from dataset."""
        futures = deque()
        
        for sample in self.dataset:
            fetch_start = time.time()
            
            # Submit decode job
            future = self.executor.submit(self._process_sample, sample)
            futures.append((future, fetch_start))
            
            # Check completed futures
            while futures and futures[0][0].done():
                future, fetch_start = futures.popleft()
                try:
                    result = future.result(timeout=self.config.decode_timeout)
                    if result is not None:
                        self.decode_queue.put(result)
                        if self.stats:
                            self.stats.total_samples += 1
                            self.stats.fetch_times.append(time.time() - fetch_start)
                    else:
                        if self.stats:
                            self.stats.failed_samples += 1
                except Exception as e:
                    print(f"Decode error: {e}")
                    if self.stats:
                        self.stats.failed_samples += 1
            
            # Prevent queue overflow
            while len(futures) > self.config.num_workers * 2:
                time.sleep(0.01)
        
        # Wait for remaining futures
        for future, fetch_start in futures:
            try:
                result = future.result(timeout=self.config.decode_timeout)
                if result is not None:
                    self.decode_queue.put(result)
            except Exception as e:
                print(f"Final decode error: {e}")
        
        # Signal completion
        self.decode_queue.put(None)
    
    def _batch_worker(self):
        """Background thread that creates batches."""
        buffer = []
        
        while True:
            # Get decoded sample
            sample = self.decode_queue.get()
            if sample is None:
                # End of data
                if buffer:
                    batch = self._create_batch(buffer)
                    self.batch_queue.put(batch)
                self.batch_queue.put(None)
                break
            
            buffer.append(sample)
            
            # Create batch when full
            if len(buffer) >= self.config.batch_size:
                batch_start = time.time()
                batch = self._create_batch(buffer[:self.config.batch_size])
                self.batch_queue.put(batch)
                buffer = buffer[self.config.batch_size:]
                
                if self.stats:
                    self.stats.batch_times.append(time.time() - batch_start)
    
    def _process_sample(self, sample: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process a single sample (runs in parallel)."""
        decode_start = time.time()
        
        try:
            audio_array = None
            sr = None
            
            # Extract audio
            if self.audio_column and self.audio_column in sample:
                audio_data = sample[self.audio_column]
                
                if isinstance(audio_data, dict) and 'array' in audio_data:
                    audio_array = audio_data['array']
                    sr = audio_data.get('sampling_rate', self.config.sample_rate)
                elif isinstance(audio_data, (np.ndarray, list)):
                    audio_array = np.array(audio_data)
                    sr = self.config.sample_rate
                elif isinstance(audio_data, bytes):
                    # MP3/compressed audio - main bottleneck
                    audio_array, sr = librosa.load(
                        io.BytesIO(audio_data), 
                        sr=self.config.sample_rate,  # Directly load at target SR
                        mono=True,  # Load as mono directly
                        res_type='kaiser_fast'  # Faster resampling
                    )
                else:
                    return None
                
                # Resample if needed (should be rare with direct loading)
                if sr != self.config.sample_rate:
                    audio_array = librosa.resample(
                        audio_array,
                        orig_sr=sr,
                        target_sr=self.config.sample_rate,
                        res_type='kaiser_fast'
                    )
            else:
                return None
            
            # Convert to mono if needed
            if audio_array.ndim > 1:
                audio_array = np.mean(audio_array, axis=0)
            
            # Truncate if too long
            if len(audio_array) > self.max_samples:
                audio_array = audio_array[:self.max_samples]
            
            # Fast normalization
            max_val = np.abs(audio_array).max()
            if max_val > 0:
                audio_array = audio_array * (0.95 / max_val)
            
            # Parse metadata
            metadata = {}
            if self.metadata_column and self.metadata_column in sample:
                try:
                    if self.metadata_column == 'json' and isinstance(sample[self.metadata_column], str):
                        metadata = json.loads(sample[self.metadata_column])
                    elif isinstance(sample[self.metadata_column], dict):
                        metadata = sample[self.metadata_column]
                    elif isinstance(sample[self.metadata_column], str):
                        metadata = {'text': sample[self.metadata_column]}
                except:
                    pass
            
            result = {
                'audio': audio_array.astype(np.float32),
                'length': len(audio_array)
            }
            
            # Add metadata
            for key, value in metadata.items():
                result[f'meta_{key}'] = value
            
            if self.stats:
                self.stats.decode_times.append(time.time() - decode_start)
            
            return result
            
        except Exception as e:
            if self.stats and self.stats.failed_samples < 10:
                print(f"Sample processing error: {e}")
            return None
    
    def _create_batch(self, samples: List[Dict[str, Any]]) -> Dict[str, jax.Array]:
        """Create a padded batch from samples."""
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
        
        # Create masks
        mask = create_padding_mask(lengths_array, max_length, causal=False)
        
        # Create encoder-level masks
        downsample_factor = 480 if self.config.sample_rate == 24000 else 960
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
        
        # Collect metadata
        metadata_keys = set()
        for sample in samples:
            metadata_keys.update(k for k in sample.keys() if k.startswith('meta_'))
        
        for key in metadata_keys:
            batch[key] = [sample.get(key, None) for sample in samples]
        
        return batch
    
    def __iter__(self) -> Iterator[Dict[str, jax.Array]]:
        """Iterate over batches."""
        batch_idx = 0
        
        while True:
            batch = self.batch_queue.get()
            if batch is None:
                break
            
            yield batch
            batch_idx += 1
            
            # Print profiling stats
            if self.stats and batch_idx % self.config.profile_interval == 0:
                self.stats.print_stats(batch_idx)
    
    def close(self):
        """Clean up resources."""
        self.executor.shutdown(wait=False)


def create_optimized_emilia_loader(
    language: str = "EN",
    split: str = "train", 
    batch_size: int = 8,
    sample_rate: int = 24000,
    num_workers: int = 4,
    prefetch_batches: int = 2,
    profile: bool = True,
    **kwargs
) -> OptimizedAudioLoader:
    """Create an optimized loader for Emilia dataset."""
    config = AudioConfig(
        dataset_name="amphion/Emilia-Dataset",
        dataset_config="default",  # Emilia uses 'default' config
        split=split,
        sample_rate=sample_rate,
        batch_size=batch_size,
        streaming=True,  # Always use streaming for large datasets
        num_workers=num_workers,
        prefetch_batches=prefetch_batches,
        profile=profile,
        **kwargs
    )
    return OptimizedAudioLoader(config)


def test_optimized_loader():
    """Test the optimized loader."""
    print("Testing optimized Emilia loader...")
    
    loader = create_optimized_emilia_loader(
        batch_size=4,
        max_duration_seconds=5.0,
        num_workers=4,
        prefetch_batches=2,
        profile=True,
        profile_interval=5
    )
    
    try:
        for i, batch in enumerate(loader):
            if i >= 20:  # Test 20 batches
                break
            
            if i % 5 == 0:
                print(f"\nBatch {i} shapes:")
                print(f"  Audio: {batch['audio'].shape}")
                print(f"  Lengths: {batch['lengths']}")
                if 'meta_text' in batch:
                    print(f"  Has text metadata: {len(batch['meta_text'])} samples")
        
        print("\nTest completed successfully!")
        
    finally:
        loader.close()


if __name__ == "__main__":
    test_optimized_loader()