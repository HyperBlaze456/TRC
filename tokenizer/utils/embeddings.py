import jax.numpy as jnp
from flax import nnx


def get_sinusoidal(t, dim, theta = 10000):
    half = dim // 2
    freqs = jnp.exp(
        -jnp.log(theta) * jnp.arange(0, half) / half
    )
    args = t[:, None] * freqs[None]
    embedding = jnp.concat([jnp.cos(args), jnp.sin(args)], axis = -1)

    return embedding


class TimeStepEmbedder(nnx.Module):
    def __init__(self, dim, hidden_size, rngs: nnx.Rngs):
        self.learnable_transformation = nnx.Sequential(
            nnx.Linear(dim, hidden_size, rngs=rngs),
            nnx.swish,
            nnx.Linear(hidden_size, hidden_size, rngs=rngs),
        )
        self.dim = dim

    def __call__(self, t):
        t_freq = get_sinusoidal(t, self.dim)
        t_emb = self.learnable_transformation(t_freq)
        return t_emb


class RotaryPositionalEmbedding(nnx.Module):
    def __init__(self, dim: int, max_seq_len: int = 5000, base: float = 10000.0):
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        # Precompute frequency bands
        inv_freq = 1.0 / (base ** (jnp.arange(0, dim, 2) / dim))
        self.inv_freq = inv_freq
        
    def __call__(self, seq_len: int, dtype=jnp.float32):
        # Generate position indices
        t = jnp.arange(seq_len, dtype=dtype)
        
        # Compute outer product of positions and frequencies
        freqs = jnp.outer(t, self.inv_freq)
        
        # Create rotation matrix elements
        cos = jnp.cos(freqs)
        sin = jnp.sin(freqs)
        
        return cos, sin
    
    def apply_rotary_pos_emb(self, x, cos, sin):
        """Apply rotary embeddings to input tensor."""
        # x shape: [batch, seq_len, heads, head_dim] or [batch, seq_len, head_dim]
        # cos, sin shape: [seq_len, head_dim//2]
        
        # Ensure cos and sin have compatible shape with x
        ndim_diff = x.ndim - cos.ndim
        if ndim_diff > 0:
            # Add dimensions to match x's shape
            for _ in range(ndim_diff - 1):  # -1 because seq_len dimension is already there
                cos = cos[None, ...]  # Add batch dimension
                sin = sin[None, ...]
            if x.ndim == 4:  # If x has head dimension
                cos = cos[:, :, None, :]  # Add head dimension
                sin = sin[:, :, None, :]
        
        # Split x into two halves along the last dimension
        x1, x2 = jnp.split(x, 2, axis=-1)
        
        # Apply rotation
        rx1 = x1 * cos - x2 * sin
        rx2 = x2 * cos + x1 * sin
        
        # Concatenate back
        return jnp.concatenate([rx1, rx2], axis=-1)


class RelativePositionalEncoding(nnx.Module):
    def __init__(self, hidden_size: int, max_len: int = 5000, rngs: nnx.Rngs = None):
        self.hidden_size = hidden_size
        self.max_len = max_len
        
        # Learnable relative position bias
        self.rel_pos_bias = nnx.Param(
            nnx.initializers.normal(stddev=0.02)(rngs(), (2 * max_len - 1, hidden_size))
        )
        
    def __call__(self, seq_len: int):
        # Generate relative positions matrix
        positions = jnp.arange(seq_len)[:, None] - jnp.arange(seq_len)[None, :]
        positions = positions + self.max_len - 1
        positions = jnp.clip(positions, 0, 2 * self.max_len - 2)
        
        # Get embeddings
        rel_pos_emb = self.rel_pos_bias[positions]
        return rel_pos_emb