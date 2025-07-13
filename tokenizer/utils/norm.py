import jax.numpy as jnp
from flax import nnx

class AdaLNZero(nnx.Module):
    def __init__(self, dim:int, rngs: nnx.Rngs, eps: float = 1e-6):
        self.norm = nnx.LayerNorm(dim, epsilon=eps, rngs=rngs)
        self.mlp = nnx.Linear(
            dim, dim*3,
            kernel_init=nnx.initializers.zeros_init(),
            bias_init=nnx.initializers.zeros_init(),
            rngs=rngs
        )

    def __call__(self, x, cond):
        scale, shift, gate = jnp.split(self.mlp(nnx.silu(cond)), 2, axis=-1)
        scale += 1 # add one to delta γ
        if x.ndim > scale.ndim:
            # Encountered B, ..., D input, add axis to match.
            for _ in range(x.ndim - scale.ndim):
                scale = jnp.expand_dims(scale, axis=1)
                shift = jnp.expand_dims(shift, axis=1)
                gate = jnp.expand_dims(gate, axis=1)

        normalized = self.norm(x)
        return normalized * scale + shift, gate # apply gate after attn/ffn pass

class AdaRMSZero(nnx.Module):
    def __init__(self, dim:int, rngs: nnx.Rngs, eps: float = 1e-6):
        self.norm = nnx.RMSNorm(dim, epsilon=eps, rngs = rngs)
        self.mlp = nnx.Linear(
            dim, dim*2,
            kernel_init=nnx.initializers.zeros_init(),
            bias_init=nnx.initializers.zeros_init(),
            rngs=rngs
        )
    def __call__(self, x, cond):
        scale, gate = jnp.split(self.mlp(nnx.silu(cond)), 2, axis=-1)
        scale += 1
        if x.ndim > scale.ndim:
            for _ in range(x.ndim - scale.ndim):
                scale = jnp.expand_dims(scale, axis=1)
                gate = jnp.expand_dims(gate, axis=1)
        normalized = self.norm(x)
        return normalized * scale, gate # no shift, but have to scale.

class WeightNorm(nnx.Module):
    pass # I have ideas... but need to see more about how nnx module works!