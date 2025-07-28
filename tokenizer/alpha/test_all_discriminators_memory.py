"""Test script to compare memory usage between original and optimized discriminators."""

import os
import sys

# Set JAX traceback filtering off for detailed error traces
os.environ["JAX_TRACEBACK_FILTERING"] = "off"

# Add the TRC directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import jax
import jax.numpy as jnp
from flax import nnx
import gc

from tokenizer.alpha.components.discriminators import (
    MultiPeriodDiscriminator,
    MultiScaleDiscriminator,
    STFTDiscriminator,
)
from tokenizer.alpha.components.discriminators.optimized_scale_discriminator import (
    OptimizedMultiScaleDiscriminator
)
from tokenizer.alpha.components.discriminators.optimized_period_discriminator import (
    OptimizedMultiPeriodDiscriminator
)
from tokenizer.alpha.components.discriminators.optimized_stft_discriminator import (
    OptimizedSTFTDiscriminator
)

def log_memory(label: str):
    """Log GPU memory usage with detailed stats."""
    for i, device in enumerate(jax.devices()):
        stats = device.memory_stats()
        if stats:
            used_gb = stats['bytes_in_use'] / (1024**3)
            limit_gb = stats.get('bytes_limit', 0) / (1024**3)
            print(f"[{label}] GPU {i}: {used_gb:.2f}GB / {limit_gb:.2f}GB")
    return stats['bytes_in_use'] if stats else 0

def get_param_stats(model):
    """Get parameter count and memory usage."""
    params = nnx.state(model, nnx.Param)
    param_count = sum(p.size for p in jax.tree.leaves(params))
    param_memory = sum(p.nbytes for p in jax.tree.leaves(params))
    return param_count, param_memory

def test_discriminator(name, model_fn, forward_fn, audio, rngs):
    """Test a single discriminator and return memory stats."""
    print("="*60)
    print(f"Testing {name}")
    print("="*60)
    
    # Initial memory
    before_create = log_memory(f"Before creating {name}")
    
    # Create model
    model = model_fn(rngs=rngs)
    after_create = log_memory(f"After creating {name}")
    create_diff = (after_create - before_create) / (1024**3)
    print(f"Memory diff for creation: +{create_diff:.2f} GB\n")
    
    # Get parameter stats
    param_count, param_memory = get_param_stats(model)
    print(f"Parameters: {param_count:,} ({param_memory / (1024**3):.3f} GB)")
    
    # JIT compile
    try:
        @nnx.jit
        def jit_forward(model, audio):
            return forward_fn(model, audio)
        
        before_jit = log_memory(f"Before {name} JIT compilation")
        
        # Force compilation
        output = jit_forward(model, audio)
        
        after_jit = log_memory(f"After {name} JIT compilation")
        jit_diff = (after_jit - before_jit) / (1024**3)
        expansion_ratio = (after_jit - before_jit) / param_memory if param_memory > 0 else 0
        
        print(f"Memory diff for JIT compilation: +{jit_diff:.2f} GB")
        print(f"Memory expansion ratio: {expansion_ratio:.1f}x\n")
        
        return {
            'name': name,
            'create_memory': create_diff,
            'jit_memory': jit_diff,
            'param_memory': param_memory / (1024**3),
            'expansion_ratio': expansion_ratio,
            'success': True
        }
        
    except Exception as e:
        print(f"{name} error: {e}\n")
        return {
            'name': name,
            'create_memory': create_diff,
            'jit_memory': 0,
            'param_memory': param_memory / (1024**3),
            'expansion_ratio': 0,
            'success': False
        }
    finally:
        # Cleanup
        del model
        if 'output' in locals():
            del output
        gc.collect()

def test_all_discriminators():
    """Test all discriminator variants and compare memory usage."""
    
    # Configuration
    batch_size = 32
    seq_length = 240000  # 10 seconds at 24kHz
    
    print("=== Discriminator Memory Usage Comparison ===")
    print(f"Batch size: {batch_size}")
    print(f"Sequence length: {seq_length} samples\n")
    
    # Initialize RNG
    rngs = nnx.Rngs(42)
    
    # Create fake audio batch
    initial_memory = log_memory("Initial state")
    
    audio = jax.random.normal(rngs(), (batch_size, seq_length, 1))
    print(f"Audio shape: {audio.shape}")
    print(f"Audio memory: {audio.nbytes / (1024**3):.2f} GB")
    
    audio_memory = log_memory("After creating audio")
    print(f"Memory diff: +{(audio_memory - initial_memory) / (1024**3):.2f} GB\n")
    
    # Test all discriminator variants
    results = []
    
    # 1. Original MSD
    result = test_discriminator(
        "Original MultiScaleDiscriminator",
        lambda rngs: MultiScaleDiscriminator(rngs=rngs),
        lambda model, audio: model(audio, training=True),
        audio, rngs
    )
    results.append(result)
    
    # 2. Optimized MSD
    result = test_discriminator(
        "Optimized MultiScaleDiscriminator",
        lambda rngs: OptimizedMultiScaleDiscriminator(rngs=rngs),
        lambda model, audio: model(audio, training=True),
        audio, rngs
    )
    results.append(result)
    
    # 3. Original MPD
    result = test_discriminator(
        "Original MultiPeriodDiscriminator",
        lambda rngs: MultiPeriodDiscriminator(rngs=rngs),
        lambda model, audio: model(audio),
        audio, rngs
    )
    results.append(result)
    
    # 4. Optimized MPD
    result = test_discriminator(
        "Optimized MultiPeriodDiscriminator",
        lambda rngs: OptimizedMultiPeriodDiscriminator(rngs=rngs),
        lambda model, audio: model(audio),
        audio, rngs
    )
    results.append(result)
    
    # 5. Original STFTD
    result = test_discriminator(
        "Original STFTDiscriminator",
        lambda rngs: STFTDiscriminator(rngs=rngs),
        lambda model, audio: model(audio, training=True),
        audio, rngs
    )
    results.append(result)
    
    # 6. Optimized STFTD
    result = test_discriminator(
        "Optimized STFTDiscriminator",
        lambda rngs: OptimizedSTFTDiscriminator(rngs=rngs),
        lambda model, audio: model(audio, training=True),
        audio, rngs
    )
    results.append(result)
    
    # Print summary
    print("="*60)
    print("SUMMARY")
    print("="*60)
    print(f"{'Discriminator':<40} {'Params':<10} {'JIT Mem':<10} {'Expansion':<10}")
    print("-"*70)
    
    for r in results:
        if r['success']:
            print(f"{r['name']:<40} {r['param_memory']:.3f} GB  {r['jit_memory']:>6.2f} GB  {r['expansion_ratio']:>6.1f}x")
        else:
            print(f"{r['name']:<40} {r['param_memory']:.3f} GB  FAILED")
    
    # Calculate improvements
    print("\n" + "="*60)
    print("MEMORY REDUCTION COMPARISON")
    print("="*60)
    
    # MSD comparison
    orig_msd = next(r for r in results if "Original MultiScale" in r['name'])
    opt_msd = next(r for r in results if "Optimized MultiScale" in r['name'])
    if orig_msd['success'] and opt_msd['success']:
        reduction = orig_msd['jit_memory'] - opt_msd['jit_memory']
        percent = (reduction / orig_msd['jit_memory']) * 100
        print(f"MSD: {reduction:.2f} GB reduction ({percent:.1f}%)")
    
    # MPD comparison
    orig_mpd = next(r for r in results if "Original MultiPeriod" in r['name'])
    opt_mpd = next(r for r in results if "Optimized MultiPeriod" in r['name'])
    if orig_mpd['success'] and opt_mpd['success']:
        reduction = orig_mpd['jit_memory'] - opt_mpd['jit_memory']
        percent = (reduction / orig_mpd['jit_memory']) * 100
        print(f"MPD: {reduction:.2f} GB reduction ({percent:.1f}%)")
    
    # STFTD comparison
    orig_stftd = next(r for r in results if "Original STFT" in r['name'])
    opt_stftd = next(r for r in results if "Optimized STFT" in r['name'])
    if orig_stftd['success'] and opt_stftd['success']:
        reduction = orig_stftd['jit_memory'] - opt_stftd['jit_memory']
        percent = (reduction / orig_stftd['jit_memory']) * 100
        print(f"STFTD: {reduction:.2f} GB reduction ({percent:.1f}%)")
    
    final_memory = log_memory("\nFinal state")
    print(f"\nTotal memory used: {final_memory / (1024**3):.2f} GB")

if __name__ == "__main__":
    test_all_discriminators()