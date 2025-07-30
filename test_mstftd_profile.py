import jax
import jax.numpy as jnp
from flax import nnx
from tokenizer.alpha_new.discriminators.mstftd import MSTFTD, STFTDiscriminator
import jax.profiler
import os
import time

# Enable XLA memory profiling
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.8'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'
# Enable memory profiling in XLA
os.environ['XLA_FLAGS'] = '--xla_hlo_profile --xla_gpu_enable_memory_profile'
# Additional profiling flags
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'  # Enable all logs
os.environ['JAX_TRACEBACK_FILTERING'] = 'off'

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
            
            # JIT compile the forward pass with memory profiling
            @nnx.jit
            def forward_pass(audio):
                return disc(audio)
            
            # Warm up JIT compilation
            print("Warming up JIT...")
            _ = forward_pass(jax.random.normal(key, shape=(1, 1000, 1)))
            
            # Run with profiling
            print("Running profiled forward pass...")
            featmaps = forward_pass(audio)
            
            # Block until computation is done
            jax.block_until_ready(featmaps)
            
            print(f"Success! Output shapes:")
            for i, feat in enumerate(featmaps):
                print(f"  Feature map {i}: {feat.shape}")
                
        except Exception as e:
            print(f"Failed with error: {type(e).__name__}: {str(e)}")

def test_mstftd():
    """Test the full MSTFTD discriminator"""
    print("\n" + "="*60)
    print("Testing full MSTFTD discriminator")
    print("="*60)
    
    key = jax.random.PRNGKey(42)
    rngs = nnx.Rngs(0)
    
    # Create MSTFTD discriminator
    mstftd = MSTFTD(rngs=rngs)
    
    # Test with different batch sizes to see memory usage
    batch_sizes = [1, 2, 4, 8]
    
    for batch_size in batch_sizes:
        print(f"\nTesting with batch size {batch_size}")
        audio = jax.random.normal(key, shape=(batch_size, 48000, 1))
        
        @nnx.jit
        def forward_mstftd(audio):
            return mstftd(audio)
        
        try:
            # Warm up if first iteration
            if batch_size == 1:
                print("Warming up JIT...")
                _ = forward_mstftd(audio)
                jax.block_until_ready(_)
            
            # Run with profiling
            outputs = forward_mstftd(audio)
            jax.block_until_ready(outputs)
            
            print(f"Success! Got {len(outputs)} output tensors")
            
        except Exception as e:
            print(f"Failed with batch size {batch_size}: {e}")
            break

if __name__ == "__main__":
    # Set up profiling with memory tracking
    print("Starting JAX profiler trace with memory profiling...")
    
    # Use context manager for cleaner profiling
    with jax.profiler.trace("./mstftd_profile", create_perfetto_link=False):
        # Test individual STFT discriminators
        test_stft_discriminator()
        
        # Test full MSTFTD
        test_mstftd()
        
        # Force some memory activity to ensure capture
        print("\nForcing memory activity...")
        for i in range(3):
            x = jax.random.normal(jax.random.PRNGKey(i), shape=(2000, 2000))
            y = jnp.dot(x, x.T)
            jax.block_until_ready(y)
            time.sleep(0.1)
    
    print("\nProfile saved to ./mstftd_profile")
    print("To view: tensorboard --logdir=./mstftd_profile")