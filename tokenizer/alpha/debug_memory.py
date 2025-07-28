"""Debug script to isolate memory usage by component."""

import os
import sys

# Set JAX traceback filtering off for detailed error traces
os.environ["JAX_TRACEBACK_FILTERING"] = "off"

# Add the TRC directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import jax
import jax.numpy as jnp
from flax import nnx

from tokenizer.alpha.components.discriminators import (
    MultiPeriodDiscriminator,
    MultiScaleDiscriminator,
    STFTDiscriminator,
)
from tokenizer.alpha.model import SpeechTokenizer

def log_memory(label: str):
    """Log GPU memory usage."""
    for i, device in enumerate(jax.devices()):
        stats = device.memory_stats()
        if stats:
            used_gb = stats['bytes_in_use'] / (1024**3)
            limit_gb = stats.get('bytes_limit', 0) / (1024**3)
            print(f"[{label}] GPU {i}: {used_gb:.2f}GB / {limit_gb:.2f}GB")

def test_component_memory():
    """Test memory usage of individual components."""
    
    # Configuration
    batch_size = 32
    seq_length = 240000  # 10 seconds at 24kHz
    hidden_size = 512
    
    print("=== Testing Component Memory Usage ===")
    print(f"Batch size: {batch_size}")
    print(f"Sequence length: {seq_length} samples")
    
    # Initialize RNG
    rngs = nnx.Rngs(42)
    
    # Test 1: Create fake audio batch
    log_memory("Initial")
    audio = jax.random.normal(rngs(), (batch_size, seq_length, 1))
    print(f"\nAudio shape: {audio.shape}")
    print(f"Audio memory: {audio.nbytes / (1024**3):.2f} GB")
    log_memory("After creating audio")
    
    # Test 2: Generator only
    print("\n--- Testing Generator ---")
    generator = SpeechTokenizer(
        hidden_size=hidden_size,
        encoder_depth=4,
        encoder_heads=8,
        phoneme_codebook_size=100,
        bsq_spherical_dim=256,
        decoder_output_48khz=False,
        rngs=rngs,
    )
    log_memory("After creating generator")
    
    # Create encoder mask
    encoder_length = seq_length // 480  # 50Hz
    encoder_mask = jnp.ones((batch_size, 1, encoder_length, encoder_length), dtype=jnp.bool_)
    
    # Test generator forward pass
    print("\nTesting generator forward pass...")
    try:
        @nnx.jit
        def generator_forward(generator, audio, mask):
            return generator(audio, mask)
        
        log_memory("Before generator JIT")
        output = generator_forward(generator, audio, encoder_mask)
        log_memory("After generator JIT")
        print("Generator forward pass successful")
    except Exception as e:
        print(f"Generator error: {e}")
        
    # Test 3: Discriminators
    print("\n--- Testing Discriminators ---")
    
    # MultiScaleDiscriminator
    print("\nTesting MultiScaleDiscriminator...")
    msd = MultiScaleDiscriminator(rngs=rngs)
    log_memory("After creating MSD")
    
    try:
        @nnx.jit
        def msd_forward(msd, audio):
            return msd(audio, training=True)
        
        log_memory("Before MSD JIT")
        msd_out = msd_forward(msd, audio)
        log_memory("After MSD JIT")
        print("MSD forward pass successful")
    except Exception as e:
        print(f"MSD error: {e}")
    
    # MultiPeriodDiscriminator
    print("\nTesting MultiPeriodDiscriminator...")
    mpd = MultiPeriodDiscriminator(rngs=rngs)
    log_memory("After creating MPD")
    
    try:
        @nnx.jit
        def mpd_forward(mpd, audio):
            return mpd(audio)
        
        log_memory("Before MPD JIT")
        mpd_out = mpd_forward(mpd, audio)
        log_memory("After MPD JIT")
        print("MPD forward pass successful")
    except Exception as e:
        print(f"MPD error: {e}")
    
    # STFTDiscriminator
    print("\nTesting STFTDiscriminator...")
    stftd = STFTDiscriminator(rngs=rngs)
    log_memory("After creating STFTD")
    
    try:
        @nnx.jit
        def stftd_forward(stftd, audio):
            return stftd(audio, training=True)
        
        log_memory("Before STFTD JIT")
        stftd_out = stftd_forward(stftd, audio)
        log_memory("After STFTD JIT")
        print("STFTD forward pass successful")
    except Exception as e:
        print(f"STFTD error: {e}")
    
    # Test 4: Combined discriminator forward passes
    print("\n--- Testing Combined Discriminator Operations ---")
    
    try:
        @nnx.jit
        def all_discriminators(msd, mpd, stftd, audio):
            msd_out, msd_feat = msd(audio, training=True)
            mpd_out, mpd_feat = mpd(audio)
            stftd_out, stftd_feat = stftd(audio, training=True)
            return msd_out + mpd_out + stftd_out
        
        log_memory("Before combined discriminators JIT")
        combined_out = all_discriminators(msd, mpd, stftd, audio)
        log_memory("After combined discriminators JIT")
        print("Combined discriminators successful")
    except Exception as e:
        print(f"Combined discriminators error: {e}")
    
    print("\n=== Memory Test Complete ===")
    log_memory("Final")

if __name__ == "__main__":
    test_component_memory()