# Memory-Efficient Discriminator Implementation Tips

## Key Changes Made:

### 1. **Static Operations**
- Replaced dynamic padding with static padding using modulo operations
- Pre-compute windows for STFT to avoid dynamic allocation
- Use tuples instead of lists for static compilation

### 2. **JIT-Friendly Patterns**
- Added `@partial(jax.jit, static_argnames=[...])` for functions with static parameters
- Avoided conditional logic in forward passes
- Pre-allocated data structures where possible

### 3. **Vectorized Operations**
- Used `jax.vmap` for frame extraction in STFT instead of explicit indexing
- This reduces memory overhead from intermediate array creation

### 4. **Memory Usage Comparison**

The original discriminators cause 100-200x memory expansion because:
- Dynamic reshaping forces JAX to compile for all possible input shapes
- List accumulation creates multiple intermediate copies
- Conditional padding creates branching in the computation graph

The optimized versions should reduce memory usage by:
- ~50-70% for PeriodDiscriminator
- ~60-80% for STFTDiscriminator

## Additional Optimization Strategies:

1. **Gradient Checkpointing**: For very deep discriminators, use `nnx.remat` to trade compute for memory:
```python
@nnx.remat
def forward_block(self, x):
    # computation here
```

2. **Mixed Precision**: Use `bfloat16` for discriminators:
```python
x = x.astype(jnp.bfloat16)
```

3. **Scan Instead of Loop**: For repeated operations:
```python
def scan_fn(carry, conv):
    x = conv(carry)
    x = nnx.leaky_relu(x, 0.1)
    return x, x

final_x, features = jax.lax.scan(scan_fn, x, self.convs)
```

4. **Compile Discriminators Separately**: Instead of JIT-compiling the entire training step, compile discriminators separately to control memory usage:
```python
@jax.jit
def discriminator_forward(discriminator, x):
    return discriminator(x)
```