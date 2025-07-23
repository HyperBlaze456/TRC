#!/usr/bin/env python3
"""
Emilia Dataset Loader Test Script
Tests various loading methods with safety checks for disk space and memory.
"""

import os
import sys
import time
import gc
import psutil
import numpy as np
import jax
import jax.numpy as jnp
from datasets import load_dataset, Audio
from dotenv import load_dotenv
from huggingface_hub import login
import warnings
from datetime import datetime

# Suppress some warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)

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

class Colors:
    """Terminal colors for pretty output"""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'


def print_header(text: str):
    """Print a formatted header"""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'=' * 60}{Colors.END}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text:^60}{Colors.END}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'=' * 60}{Colors.END}")


def print_success(text: str):
    """Print success message"""
    print(f"{Colors.GREEN}✓ {text}{Colors.END}")


def print_warning(text: str):
    """Print warning message"""
    print(f"{Colors.WARNING}⚠ {text}{Colors.END}")


def print_error(text: str):
    """Print error message"""
    print(f"{Colors.FAIL}✗ {text}{Colors.END}")


def print_info(text: str):
    """Print info message"""
    print(f"{Colors.CYAN}ℹ {text}{Colors.END}")


def get_system_info():
    """Get system information"""
    print_header("System Information")

    # CPU info
    cpu_count = psutil.cpu_count()
    cpu_percent = psutil.cpu_percent(interval=1)
    print_info(f"CPU: {cpu_count} cores, {cpu_percent}% usage")

    # Memory info
    memory = psutil.virtual_memory()
    print_info(f"RAM: {memory.total / (1024 ** 3):.1f} GB total, "
               f"{memory.available / (1024 ** 3):.1f} GB available ({memory.percent}% used)")

    # Disk info
    disk = psutil.disk_usage('/')
    print_info(f"Disk: {disk.total / (1024 ** 3):.1f} GB total, "
               f"{disk.free / (1024 ** 3):.1f} GB free ({disk.percent}% used)")

    # JAX info
    print_info(f"JAX version: {jax.__version__}")
    print_info(f"JAX devices: {jax.devices()}")

    # HuggingFace cache info
    cache_dir = os.path.expanduser("~/.cache/huggingface")
    if os.path.exists(cache_dir):
        cache_size = sum(
            os.path.getsize(os.path.join(dirpath, filename))
            for dirpath, dirnames, filenames in os.walk(cache_dir)
            for filename in filenames
        ) / (1024 ** 3)
        print_info(f"HuggingFace cache size: {cache_size:.2f} GB")


def test_streaming_basic():
    """Test basic streaming functionality"""
    print_header("Test 1: Basic Streaming")

    try:
        start_time = time.time()

        print_info("Loading dataset with streaming=True...")
        dataset = load_dataset(
            "amphion/Emilia-Dataset",
            split="train",
            streaming=True,
        )

        print_info("Casting audio column to 16kHz...")
        dataset = dataset.cast_column("mp3", Audio(sampling_rate=16000))

        print_info("Loading first sample...")
        first_sample = next(iter(dataset))

        # Check audio data
        if 'mp3' in first_sample:
            audio_data = first_sample['mp3']
            if isinstance(audio_data, dict) and 'array' in audio_data:
                audio_array = audio_data['array']
                print_success(f"Audio shape: {audio_array.shape}")
                print_success(f"Audio dtype: {audio_array.dtype}")
                print_success(f"Audio range: [{audio_array.min():.3f}, {audio_array.max():.3f}]")
                print_success(f"Sample rate: {audio_data.get('sampling_rate', 'N/A')} Hz")
            else:
                print_error(f"Unexpected audio format: {type(audio_data)}")

        # Check other fields
        print_info("Available fields in dataset:")
        for key in first_sample.keys():
            if key != 'mp3':
                value = first_sample[key]
                if isinstance(value, str):
                    preview = value[:50] + "..." if len(value) > 50 else value
                    print_info(f"  - {key}: {preview}")
                else:
                    print_info(f"  - {key}: {type(value).__name__}")

        elapsed = time.time() - start_time
        print_success(f"Basic streaming test completed in {elapsed:.2f} seconds")

    except Exception as e:
        print_error(f"Basic streaming test failed: {str(e)}")
        import traceback
        traceback.print_exc()


def test_jax_conversion():
    """Test conversion to JAX arrays"""
    print_header("Test 2: JAX Array Conversion")

    try:
        dataset = load_dataset(
            "amphion/Emilia-Dataset",
            split="train",
            streaming=True,
            
        )
        dataset = dataset.cast_column("mp3", Audio(sampling_rate=16000))

        print_info("Converting to JAX array...")
        first_sample = next(iter(dataset))
        audio_numpy = first_sample['mp3']['array']

        # Test JAX conversion
        start_time = time.time()
        audio_jax = jnp.array(audio_numpy)
        conversion_time = time.time() - start_time

        print_success(f"JAX array shape: {audio_jax.shape}")
        print_success(f"JAX array dtype: {audio_jax.dtype}")
        print_success(f"Conversion time: {conversion_time * 1000:.2f} ms")
        print_success(f"Array on device: {audio_jax.device}")

        # Test some JAX operations
        print_info("Testing JAX operations...")
        mean_val = jnp.mean(audio_jax)
        std_val = jnp.std(audio_jax)
        print_success(f"Mean: {mean_val:.6f}, Std: {std_val:.6f}")

    except Exception as e:
        print_error(f"JAX conversion test failed: {str(e)}")


def test_batch_loading(batch_size: int = 4, num_batches: int = 2):
    """Test batch loading with memory monitoring"""
    print_header("Test 3: Batch Loading")

    try:
        dataset = load_dataset(
            "amphion/Emilia-Dataset",
            split="train",
            streaming=True,
            
        )
        dataset = dataset.cast_column("mp3", Audio(sampling_rate=24000))

        print_info(f"Loading {num_batches} batches of size {batch_size}...")

        initial_memory = psutil.Process().memory_info().rss / (1024 ** 2)  # MB

        batch_times = []
        for batch_idx in range(num_batches):
            batch_start = time.time()
            batch = []

            for i, sample in enumerate(dataset.skip(batch_idx * batch_size).take(batch_size)):
                audio = sample['mp3']['array']
                batch.append(jnp.array(audio))


            # Stack into batch (with padding if needed)
            if batch:
                max_len = max(len(a) for a in batch)
                padded_batch = []
                for audio in batch:
                    if len(audio) < max_len:
                        audio = jnp.pad(audio, (0, max_len - len(audio)))
                    padded_batch.append(audio)

                batch_array = jnp.stack(padded_batch)
                batch_time = time.time() - batch_start
                batch_times.append(batch_time)

                current_memory = psutil.Process().memory_info().rss / (1024 ** 2)
                memory_increase = current_memory - initial_memory

                print_success(f"Batch {batch_idx + 1}: shape={batch_array.shape}, "
                              f"time={batch_time:.2f}s, memory +{memory_increase:.1f} MB")

                # Clean up
                del batch, batch_array
                gc.collect()

        avg_time = np.mean(batch_times)
        print_success(f"Average batch loading time: {avg_time:.2f} seconds")

    except Exception as e:
        print_error(f"Batch loading test failed: {str(e)}")


def test_memory_efficiency(num_samples: int = 10):
    """Test memory efficiency of streaming"""
    print_header("Test 4: Memory Efficiency")

    try:
        initial_memory = psutil.Process().memory_info().rss / (1024 ** 2)
        print_info(f"Initial memory usage: {initial_memory:.1f} MB")

        dataset = load_dataset(
            "amphion/Emilia-Dataset",
            split="train",
            streaming=True,
            
        )
        dataset = dataset.cast_column("mp3", Audio(sampling_rate=16000))

        max_memory = initial_memory
        audio_lengths = []

        print_info(f"Processing {num_samples} samples...")
        for i, sample in enumerate(dataset.take(num_samples)):
            audio = sample['mp3']['array']
            audio_jax = jnp.array(audio)
            audio_lengths.append(len(audio))

            # Check memory
            current_memory = psutil.Process().memory_info().rss / (1024 ** 2)
            max_memory = max(max_memory, current_memory)

            # Clean up immediately
            del audio, audio_jax
            gc.collect()

            if (i + 1) % 5 == 0:
                print_info(f"Processed {i + 1}/{num_samples} samples")

        final_memory = psutil.Process().memory_info().rss / (1024 ** 2)
        peak_increase = max_memory - initial_memory
        final_increase = final_memory - initial_memory

        print_success(f"Memory usage - Peak increase: {peak_increase:.1f} MB, "
                      f"Final increase: {final_increase:.1f} MB")
        print_success(f"Average audio length: {np.mean(audio_lengths) / 16000:.1f} seconds")

    except Exception as e:
        print_error(f"Memory efficiency test failed: {str(e)}")


def test_error_handling():
    """Test error handling and edge cases"""
    print_header("Test 5: Error Handling")

    # Test with invalid sampling rate
    try:
        print_info("Testing with very high sampling rate...")
        dataset = load_dataset(
            "amphion/Emilia-Dataset",
            split="train",
            streaming=True,
            
        )
        dataset = dataset.cast_column("mp3", Audio(sampling_rate=96000))
        sample = next(iter(dataset))
        print_warning("High sampling rate handled successfully")
    except Exception as e:
        print_info(f"High sampling rate error (expected): {type(e).__name__}")

    # Test disk space check
    try:
        print_info("Testing disk space check...")
        disk = psutil.disk_usage('/')
        free_gb = disk.free / (1024 ** 3)
        if free_gb < 5:
            print_warning(f"Low disk space: {free_gb:.1f} GB free")
        else:
            print_success(f"Adequate disk space: {free_gb:.1f} GB free")
    except Exception as e:
        print_error(f"Disk space check failed: {str(e)}")


def test_performance_benchmark():
    """Benchmark loading performance"""
    print_header("Test 6: Performance Benchmark")

    try:
        dataset = load_dataset(
            "amphion/Emilia-Dataset",
            split="train",
            streaming=True,
            
        )
        dataset = dataset.cast_column("mp3", Audio(sampling_rate=16000))

        # Benchmark single sample loading
        print_info("Benchmarking single sample loading...")
        times = []
        for i in range(5):
            start = time.time()
            sample = next(iter(dataset.skip(i)))
            audio = jnp.array(sample['mp3']['array'])
            elapsed = time.time() - start
            times.append(elapsed)
            print_info(f"Sample {i + 1}: {elapsed:.3f}s ({len(audio) / 16000:.1f}s audio)")

        avg_time = np.mean(times)
        std_time = np.std(times)
        print_success(f"Average loading time: {avg_time:.3f} ± {std_time:.3f} seconds")

        # Test throughput
        print_info("Testing throughput...")
        start = time.time()
        total_audio_duration = 0
        for i, sample in enumerate(dataset.take(10)):
            audio_duration = len(sample['mp3']['array']) / 16000
            total_audio_duration += audio_duration

        elapsed = time.time() - start
        throughput = total_audio_duration / elapsed
        print_success(f"Throughput: {throughput:.2f}x realtime")

    except Exception as e:
        print_error(f"Performance benchmark failed: {str(e)}")


def run_all_tests():
    """Run all tests"""
    print(f"\n{Colors.BOLD}Emilia Dataset Loader Test Suite{Colors.END}")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Get system info first
    get_system_info()

    # Run tests
    tests = [
        ("Basic Streaming", test_streaming_basic),
        ("JAX Conversion", test_jax_conversion),
        ("Batch Loading", lambda: test_batch_loading(batch_size=32, num_batches=4)),
        ("Memory Efficiency", lambda: test_memory_efficiency(num_samples=5)),
        ("Error Handling", test_error_handling),
        ("Performance Benchmark", test_performance_benchmark)
    ]

    passed = 0
    failed = 0

    for test_name, test_func in tests:
        try:
            test_func()
            passed += 1
        except KeyboardInterrupt:
            print_warning(f"\n{test_name} interrupted by user")
            break
        except Exception as e:
            print_error(f"{test_name} failed with unexpected error: {e}")
            failed += 1

    # Summary
    print_header("Test Summary")
    print_success(f"Passed: {passed}/{len(tests)}")
    if failed > 0:
        print_error(f"Failed: {failed}/{len(tests)}")

    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    # Check for command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "--quick":
            print("Running quick test only...")
            test_streaming_basic()
        elif sys.argv[1] == "--help":
            print("Usage: python test_emilia_loader.py [--quick|--help]")
            print("  --quick: Run only the basic streaming test")
            print("  --help: Show this help message")
        else:
            run_all_tests()
    else:
        run_all_tests()