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
    
    # Try to compile and get memory estimates
    print("\n" + "="*60)
    print("4. Compilation Memory Estimates")
    print("="*60)
    
    # Test smaller sizes to see scaling
    for test_seq_len in [6000, 12000, 24000]:
        print(f"\nTesting with sequence length: {test_seq_len}")
        test_audio = jax.random.normal(rngs(), (2, test_seq_len, 1))
        
        # MSD memory estimate
        try:
            compiled_msd = jax.jit(lambda s, a: msd_pure(s, a)).lower(state, test_audio)
            print(f"  MSD compilation successful")
            
            # Try to get memory estimate from compiled function
            hlo_module = compiled_msd.compile()
            hlo_text = hlo_module.as_text()
            
            # Count large allocations in HLO
            large_allocs = 0
            for line in hlo_text.split('\n'):
                if 'f32[' in line:
                    # Extract dimensions and estimate size
                    import re
                    shapes = re.findall(r'f32\[([^\]]+)\]', line)
                    for shape in shapes:
                        dims = [int(d.strip()) for d in shape.split(',') if d.strip().isdigit()]
                        if dims:
                            size = 1
                            for d in dims:
                                size *= d
                            if size > 1_000_000:  # More than 1M elements
                                large_allocs += 1
            
            print(f"  MSD large allocations (>1M elements): {large_allocs}")
            
        except Exception as e:
            print(f"  MSD compilation failed: {str(e)[:100]}...")
    
    print("\n" + "="*60)
    print("Analysis complete! Check the generated .txt files for detailed JAXpr.")
    print("="*60)

if __name__ == "__main__":
    analyze_discriminator_compilation()