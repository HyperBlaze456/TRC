"""
Memory profiling utilities for JAX training to identify OOM issues during JIT compilation.
"""

import functools
import os
import traceback
from typing import Callable, Any

import jax
import jax.numpy as jnp
from jax.experimental import profiler as jax_profiler


def enable_memory_profiling():
    """Enable comprehensive memory profiling for JAX."""
    # Enable XLA memory profiling
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".95"  # Use 95% of GPU memory
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"  # Don't preallocate memory
    
    # Enable detailed compilation logging
    jax.config.update("jax_log_compiles", True)
    jax.config.update("jax_debug_nans", True)
    
    # Enable memory tracking
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"  # Show all logs
    os.environ["XLA_FLAGS"] = (
        "--xla_gpu_strict_conv_algorithm_picker=false "
        "--xla_force_host_platform_device_count=1 "
        "--xla_dump_to=/tmp/xla_dump "
        "--xla_dump_hlo_as_text "
        "--xla_dump_hlo_as_proto "
        "--xla_dump_hlo_snapshots"
    )
    
    print("Memory profiling enabled with settings:")
    print(f"  - XLA memory fraction: 95%")
    print(f"  - Preallocate disabled")
    print(f"  - Compilation logging enabled")
    print(f"  - XLA dumps will be saved to /tmp/xla_dump")


def profile_memory_usage():
    """Print current memory usage."""
    try:
        for device in jax.devices():
            stats = device.memory_stats()
            if stats:
                print(f"\nDevice {device}:")
                print(f"  - Peak memory: {stats.get('peak_bytes_in_use', 0) / 1e9:.2f} GB")
                print(f"  - Current memory: {stats.get('bytes_in_use', 0) / 1e9:.2f} GB")
                print(f"  - Limit: {stats.get('bytes_limit', 0) / 1e9:.2f} GB")
    except Exception as e:
        print(f"Could not get memory stats: {e}")


def trace_compilation(func_name: str):
    """Decorator to trace function compilation and memory usage."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            print(f"\n{'='*60}")
            print(f"Compiling function: {func_name}")
            print(f"Function: {func.__name__}")
            print(f"Call stack:")
            for line in traceback.format_stack()[:-1]:
                print(f"  {line.strip()}")
            
            # Memory before
            print("\nMemory before compilation:")
            profile_memory_usage()
            
            try:
                # Compile and run
                result = func(*args, **kwargs)
                
                # Memory after
                print(f"\nMemory after compilation of {func_name}:")
                profile_memory_usage()
                
                return result
            except Exception as e:
                print(f"\nERROR during compilation of {func_name}: {e}")
                profile_memory_usage()
                raise
            finally:
                print(f"{'='*60}\n")
        
        return wrapper
    return decorator


def wrap_jit_with_profiling(func: Callable, static_argnums=None, static_argnames=None, **kwargs) -> Callable:
    """Wrap jax.jit with memory profiling."""
    func_name = func.__name__ if hasattr(func, '__name__') else str(func)
    
    # First wrap with our tracer
    traced_func = trace_compilation(f"JIT: {func_name}")(func)
    
    # Then apply JIT
    return jax.jit(traced_func, static_argnums=static_argnums, static_argnames=static_argnames, **kwargs)


def analyze_pytree_memory(name: str, pytree: Any):
    """Analyze memory usage of a pytree."""
    def get_size(arr):
        if isinstance(arr, jnp.ndarray):
            return arr.nbytes / 1e6  # MB
        return 0
    
    total_size = jax.tree_util.tree_reduce(
        lambda acc, x: acc + get_size(x),
        pytree,
        0
    )
    
    print(f"\n{name} PyTree Memory Analysis:")
    print(f"  Total size: {total_size:.2f} MB")
    
    # Analyze individual components
    leaves, treedef = jax.tree_util.tree_flatten(pytree)
    large_arrays = []
    
    for i, leaf in enumerate(leaves):
        if isinstance(leaf, jnp.ndarray):
            size_mb = leaf.nbytes / 1e6
            if size_mb > 1:  # Only show arrays > 1MB
                large_arrays.append((i, leaf.shape, leaf.dtype, size_mb))
    
    if large_arrays:
        print(f"  Large arrays (>1MB):")
        for i, shape, dtype, size in sorted(large_arrays, key=lambda x: x[3], reverse=True)[:10]:
            print(f"    [{i}] shape={shape}, dtype={dtype}, size={size:.2f} MB")


def profile_function_memory(func: Callable, *args, **kwargs):
    """Profile memory usage of a specific function call."""
    print(f"\nProfiling function: {func.__name__ if hasattr(func, '__name__') else 'anonymous'}")
    
    # Initial memory
    print("Initial memory state:")
    profile_memory_usage()
    
    # Create a traced version
    @trace_compilation(f"Direct call: {func.__name__ if hasattr(func, '__name__') else 'anonymous'}")
    def traced_func(*a, **kw):
        return func(*a, **kw)
    
    # Run with profiling
    try:
        result = traced_func(*args, **kwargs)
        return result
    except Exception as e:
        print(f"Error during profiled execution: {e}")
        raise


def check_model_memory(model, dummy_input_shape=(1, 1000, 1)):
    """Check memory usage of a model with dummy input."""
    print(f"\nChecking model memory: {type(model).__name__}")
    
    # Create dummy input
    dummy_input = jnp.ones(dummy_input_shape)
    
    # Analyze model parameters
    params = jax.tree_util.tree_map(lambda x: x if isinstance(x, jnp.ndarray) else None, model)
    analyze_pytree_memory(f"{type(model).__name__} parameters", params)
    
    # Try forward pass
    try:
        print(f"\nAttempting forward pass with input shape {dummy_input_shape}")
        profile_memory_usage()
        
        if hasattr(model, '__call__'):
            # For nnx models, just call directly
            output = model(dummy_input)
        else:
            print("Model doesn't have __call__ method")
            return
            
        print(f"Forward pass successful, output shape: {output.shape if hasattr(output, 'shape') else 'N/A'}")
        profile_memory_usage()
        
    except Exception as e:
        print(f"Forward pass failed: {e}")
        profile_memory_usage()