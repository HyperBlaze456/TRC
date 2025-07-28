"""Analyze JAX compilation of discriminators to understand memory usage."""

import os
import sys

# Set XLA flags
os.environ["XLA_FLAGS"] = "--xla_gpu_strict_conv_algorithm_picker=false --xla_gpu_autotune_level=0"
os.environ["JAX_TRACEBACK_FILTERING"] = "off"

# Add the TRC directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import jax
import jax.numpy as jnp
from flax import nnx
from jax import make_jaxpr

from tokenizer.alpha.components.discriminators import (
    MultiPeriodDiscriminator,
    MultiScaleDiscriminator,
    STFTDiscriminator,
)
from tokenizer.alpha.model import SpeechTokenizer

def get_gpu_memory():
    """Get current GPU memory usage in bytes."""
    for device in jax.devices():
        stats = device.memory_stats()
        if stats:
            return stats['bytes_in_use']
    return 0

def log_memory(label):
    """Log GPU memory with detailed stats."""
    for device in jax.devices():
        stats = device.memory_stats()
        if stats:
            used_gb = stats['bytes_in_use'] / (1024**3)
            print(f"[{label}] GPU memory: {used_gb:.3f} GB")
    return get_gpu_memory()

def analyze_discriminator_compilation():
    """Analyze the compilation of each discriminator."""
    
    # Smaller test size for analysis
    batch_size = 2
    seq_length = 24000  # 1 second at 24kHz
    
    print("=== Discriminator Compilation Analysis ===")
    print(f"Batch size: {batch_size}")
    print(f"Sequence length: {seq_length} samples")
    print(f"XLA_FLAGS: {os.environ.get('XLA_FLAGS', 'Not set')}\n")
    
    # Initialize
    rngs = nnx.Rngs(42)
    audio = jax.random.normal(rngs(), (batch_size, seq_length, 1))
    
    # Track initial memory
    initial_memory = log_memory("Initial")
    
    # 1. Analyze MSD
    print("="*60)
    print("1. MultiScaleDiscriminator Analysis")
    print("="*60)
    
    msd = MultiScaleDiscriminator(rngs=rngs)
    
    # Extract graphdef and state for pure function
    graphdef, state = nnx.split(msd)
    
    def msd_pure(state, audio):
        model = nnx.merge(graphdef, state)
        return model(audio, training=True)
    
    print("\nCreating JAXpr for MSD...")
    jaxpr = make_jaxpr(msd_pure)(state, audio)
    
    # Save detailed JAXpr
    with open('msd_jaxpr_detailed.txt', 'w') as f:
        f.write(str(jaxpr))
    
    # Analyze operations and memory
    op_counts = {}
    conv_ops = []
    memory_by_op = {}
    total_memory_bytes = 0
    
    # Extract shape information from JAXpr
    def get_shape_from_var(var):
        """Extract shape from a JAX variable."""
        if hasattr(var, 'aval') and hasattr(var.aval, 'shape'):
            return var.aval.shape
        return None
    
    def estimate_memory(shape, dtype='float32'):
        """Estimate memory in bytes for a tensor."""
        if shape is None:
            return 0
        elements = 1
        for dim in shape:
            elements *= dim
        bytes_per_element = 4 if dtype == 'float32' else 2  # f32=4, f16/bf16=2
        return elements * bytes_per_element
    
    # Analyze each operation
    for i, eqn in enumerate(jaxpr.eqns):
        op_name = eqn.primitive.name
        op_counts[op_name] = op_counts.get(op_name, 0) + 1
        
        # Calculate output memory for this operation
        output_memory = 0
        for outvar in eqn.outvars:
            shape = get_shape_from_var(outvar)
            if shape:
                mem = estimate_memory(shape)
                output_memory += mem
                total_memory_bytes += mem
        
        if op_name not in memory_by_op:
            memory_by_op[op_name] = 0
        memory_by_op[op_name] += output_memory
        
        if 'conv' in op_name:
            conv_ops.append((i, eqn, output_memory))
    
    print(f"\nTotal operations: {len(jaxpr.eqns)}")
    print(f"Total intermediate memory estimate: {total_memory_bytes / (1024**3):.2f} GB")
    
    print("\nOperation counts:")
    for op, count in sorted(op_counts.items(), key=lambda x: x[1], reverse=True)[:15]:
        print(f"  {op}: {count}")
    
    print("\nMemory usage by operation type:")
    for op, mem in sorted(memory_by_op.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {op}: {mem / (1024**3):.3f} GB")
    
    print(f"\nConvolution operations: {len(conv_ops)}")
    print("\nConvolution memory details:")
    conv_total_memory = 0
    for i, (idx, eqn, mem) in enumerate(conv_ops[:10]):
        conv_total_memory += mem
        print(f"\n  Conv {i+1} (equation {idx}):")
        print(f"    Primitive: {eqn.primitive.name}")
        print(f"    Output memory: {mem / (1024**2):.2f} MB")
        
        # Show output shapes
        for outvar in eqn.outvars:
            shape = get_shape_from_var(outvar)
            if shape:
                print(f"    Output shape: {shape}")
        
        # Show params if available
        if hasattr(eqn, 'params'):
            relevant_params = {k: v for k, v in eqn.params.items() 
                             if k in ['window_strides', 'padding', 'feature_group_count']}
            if relevant_params:
                print(f"    Params: {relevant_params}")
    
    print(f"\nTotal convolution output memory: {conv_total_memory / (1024**3):.2f} GB")
    
    # 2. Analyze MPD
    print("\n" + "="*60)
    print("2. MultiPeriodDiscriminator Analysis")
    print("="*60)
    
    mpd = MultiPeriodDiscriminator(rngs=rngs)
    
    # Extract graphdef and state
    graphdef_mpd, state_mpd = nnx.split(mpd)
    
    def mpd_pure(state, audio):
        model = nnx.merge(graphdef_mpd, state)
        return model(audio)
    
    print("\nCreating JAXpr for MPD...")
    jaxpr_mpd = make_jaxpr(mpd_pure)(state_mpd, audio)
    
    # Save detailed JAXpr
    with open('mpd_jaxpr_detailed.txt', 'w') as f:
        f.write(str(jaxpr_mpd))
    
    # Analyze operations
    op_counts_mpd = {}
    reshape_ops = []
    for i, eqn in enumerate(jaxpr_mpd.eqns):
        op_name = eqn.primitive.name
        op_counts_mpd[op_name] = op_counts_mpd.get(op_name, 0) + 1
        
        if 'reshape' in op_name:
            reshape_ops.append((i, eqn))
    
    print(f"\nTotal operations: {len(jaxpr_mpd.eqns)}")
    print("\nOperation counts:")
    for op, count in sorted(op_counts_mpd.items(), key=lambda x: x[1], reverse=True)[:15]:
        print(f"  {op}: {count}")
    
    print(f"\nReshape operations: {len(reshape_ops)}")
    
    # 3. Analyze STFTD
    print("\n" + "="*60)
    print("3. STFTDiscriminator Analysis")
    print("="*60)
    
    stftd = STFTDiscriminator(rngs=rngs)
    
    # Extract graphdef and state
    graphdef_stftd, state_stftd = nnx.split(stftd)
    
    def stftd_pure(state, audio):
        model = nnx.merge(graphdef_stftd, state)
        return model(audio, training=True)
    
    print("\nCreating JAXpr for STFTD...")
    jaxpr_stftd = make_jaxpr(stftd_pure)(state_stftd, audio)
    
    # Save detailed JAXpr
    with open('stftd_jaxpr_detailed.txt', 'w') as f:
        f.write(str(jaxpr_stftd))
    
    # Analyze operations
    op_counts_stftd = {}
    fft_ops = []
    for i, eqn in enumerate(jaxpr_stftd.eqns):
        op_name = eqn.primitive.name
        op_counts_stftd[op_name] = op_counts_stftd.get(op_name, 0) + 1
        
        if 'fft' in op_name:
            fft_ops.append((i, eqn))
    
    print(f"\nTotal operations: {len(jaxpr_stftd.eqns)}")
    print("\nOperation counts:")
    for op, count in sorted(op_counts_stftd.items(), key=lambda x: x[1], reverse=True)[:15]:
        print(f"  {op}: {count}")
    
    print(f"\nFFT operations: {len(fft_ops)}")
    
    # 4. Analyze SpeechTokenizer
    print("\n" + "="*60)
    print("4. SpeechTokenizer Analysis")
    print("="*60)
    
    hidden_size = 512
    generator = SpeechTokenizer(
        hidden_size=hidden_size,
        encoder_depth=4,
        encoder_heads=8,
        phoneme_codebook_size=100,
        bsq_spherical_dim=256,
        decoder_output_48khz=False,
        rngs=rngs,
    )
    
    # Create encoder mask
    encoder_length = seq_length // 480  # 50Hz
    encoder_mask = jnp.ones((batch_size, 1, encoder_length, encoder_length), dtype=jnp.bool_)
    
    # Extract graphdef and state
    graphdef_gen, state_gen = nnx.split(generator)
    
    def generator_pure(state, audio, mask):
        model = nnx.merge(graphdef_gen, state)
        return model(audio, mask)
    
    print("\nCreating JAXpr for SpeechTokenizer...")
    jaxpr_gen = make_jaxpr(generator_pure)(state_gen, audio, encoder_mask)
    
    # Count operations
    op_counts_gen = {}
    total_gen_memory = 0
    for i, eqn in enumerate(jaxpr_gen.eqns):
        op_name = eqn.primitive.name
        op_counts_gen[op_name] = op_counts_gen.get(op_name, 0) + 1
        
        # Calculate memory
        for outvar in eqn.outvars:
            shape = get_shape_from_var(outvar)
            if shape:
                total_gen_memory += estimate_memory(shape)
    
    print(f"\nTotal operations: {len(jaxpr_gen.eqns)}")
    print(f"Total intermediate memory estimate: {total_gen_memory / (1024**3):.2f} GB")
    
    print("\nTop operation counts:")
    for op, count in sorted(op_counts_gen.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {op}: {count}")
    
    # 5. Compare JAXpr estimates with actual GPU memory
    print("\n" + "="*60)
    print("5. JAXpr vs Actual GPU Memory Comparison")
    print("="*60)
    
    # Test compilation with actual memory tracking
    print("\nTesting actual compilation and memory usage...")
    
    # Define test configurations
    models = [
        ("SpeechTokenizer", generator_pure, (state_gen, audio, encoder_mask), total_gen_memory),
        ("MSD", msd_pure, (state, audio), total_memory_bytes),
        ("MPD", mpd_pure, (state_mpd, audio), None),  # We didn't calculate MPD memory
        ("STFTD", stftd_pure, (state_stftd, audio), None),  # We didn't calculate STFTD memory
    ]
    
    print(f"\n{'Model':<20} {'JAXpr Est.':<15} {'Pre-JIT':<15} {'Post-JIT':<15} {'JIT Overhead':<15} {'Est. vs Actual'}")
    print("-" * 95)
    
    for model_name, pure_fn, args, jaxpr_estimate in models:
        # Measure memory before JIT
        import gc
        gc.collect()
        pre_jit_memory = get_gpu_memory()
        
        try:
            # JIT compile
            jitted_fn = jax.jit(pure_fn)
            
            # First call triggers compilation
            _ = jitted_fn(*args)
            
            # Measure memory after JIT
            post_jit_memory = get_gpu_memory()
            
            # Calculate differences
            jit_overhead_bytes = post_jit_memory - pre_jit_memory
            jit_overhead_gb = jit_overhead_bytes / (1024**3)
            
            if jaxpr_estimate:
                jaxpr_gb = jaxpr_estimate / (1024**3)
                ratio = jit_overhead_gb / jaxpr_gb if jaxpr_gb > 0 else 0
                est_str = f"{jaxpr_gb:.2f} GB"
                ratio_str = f"{ratio:.1f}x"
            else:
                est_str = "N/A"
                ratio_str = "N/A"
            
            print(f"{model_name:<20} {est_str:<15} {pre_jit_memory/(1024**3):.2f} GB{'':<7} "
                  f"{post_jit_memory/(1024**3):.2f} GB{'':<7} {jit_overhead_gb:.2f} GB{'':<7} {ratio_str}")
            
        except Exception as e:
            print(f"{model_name:<20} {'ERROR: ' + str(e)[:70]}")
    
    # Final memory state
    final_memory = log_memory("\nFinal")
    print(f"\nTotal memory increase from start: {(final_memory - initial_memory) / (1024**3):.2f} GB")
    
    print("\n" + "="*60)
    print("Analysis complete! Key findings:")
    print("- JAXpr estimates show theoretical memory for intermediate values")
    print("- Actual JIT compilation uses significantly more memory due to:")
    print("  * XLA optimization passes")
    print("  * Convolution algorithm workspace")
    print("  * Gradient tape (if training)")
    print("  * Memory alignment and padding")
    print("="*60)

if __name__ == "__main__":
    analyze_discriminator_compilation()