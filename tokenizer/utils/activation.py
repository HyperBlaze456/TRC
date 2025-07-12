from flax import nnx
import jax.numpy as jnp

class Snake(nnx.Module):
    def __init__(self, dim: int, alpha:float = 1.0, alpha_trainable = True, alpha_logscale = False):
        self.dim = dim
        self.alpha_logscale = alpha_logscale
        if self.alpha_logscale:
            self.alpha = nnx.Param(jnp.zeros(dim) * alpha)
        else:
            self.alpha = nnx.Param(jnp.ones(dim) * alpha)

        self.epsilon = 1e-9
    def __call__(self, x):
        alpha = jnp.expand_dims(self.alpha, axis=(0, 1)) # B, N, D.
        if self.alpha_logscale:
            alpha = jnp.exp(alpha)
        x = x + (1.0 / (alpha + self.epsilon)) * jnp.pow(jnp.sin(x * alpha), 2)
        return x