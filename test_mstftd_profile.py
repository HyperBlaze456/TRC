import jax
import jax.numpy as jnp
from flax import nnx
from tokenizer.alpha_new.discriminators.mstftd import MSTFTD, STFTDiscriminator
import jax.profiler

# Enable memory profiling
import os
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.8'  # Use only 80% of GPU memory
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'  # Don't preallocate memory

def test_stft_discriminator():
    key = jax.random.PRNGKey(42)
    rngs = nnx.Rngs(0)
    
    # Test each FFT size individually
    configs = [
        (2048, 512, 2048, "fft_2048"),
        (1024, 256, 1024, "fft_1024"),
        (512, 128, 512, "fft_512"),
    ]
    
    for fft_size, hop_length, win_length, name in configs:
        print(f"\n{'='*50}")
        print(f"Testing {name}: fft_size={fft_size}, hop_length={hop_length}")
        print(f"{'='*50}")
        
        # Use step trace annotation instead of creating new trace
        with jax.profiler.StepTraceAnnotation(f"test_{name}"):
            try:
                # Create discriminator
                disc = STFTDiscriminator(
                    fft_size=fft_size,
                    hop_length=hop_length,
                    win_length=win_length,
                    rngs=rngs
                )
                
                # Create test input
                audio = jax.random.normal(key, shape=(32, 168_000, 1))
                
                # Log device memory before computation
                for device in jax.devices():
                    print(f"Device {device} memory stats before:")
                    print(device.memory_stats())
                
                # Run discriminator
                with jax.profiler.StepTraceAnnotation(f"forward_{name}"):
                    featmaps = disc(audio)
                
                print(f"Success! Output shapes:")
                for i, feat in enumerate(featmaps):
                    print(f"  Feature map {i}: {feat.shape}")
                
            except Exception as e:
                print(f"Failed with error: {type(e).__name__}: {str(e)}")
                # Memory stats might still be available
                for device in jax.devices():
                    print(f"Device {device} memory stats after error:")
                    print(device.memory_stats())

if __name__ == "__main__":
    # Don't start a new trace if one is already running
    # Just run the test with annotations
    test_stft_discriminator()
    
    print("\nDebug prints will show shapes even if OOM occurs")
    print("Profile data will be in the parent trace")