from flax import nnx
import jax
import jax.numpy as jnp
from functools import partial


class OptimizedPeriodDiscriminator(nnx.Module):
    """Memory-efficient period discriminator with JIT-friendly operations."""

    def __init__(self, period: int, kernel_size: int = 5, stride: int = 3, rngs: nnx.Rngs = None):
        self.period = period

        self.convs = [
            nnx.Conv(1, 32, kernel_size=(kernel_size, 1), strides=(stride, 1), padding=((2, 2), (0, 0)), rngs=rngs),
            nnx.Conv(32, 128, kernel_size=(kernel_size, 1), strides=(stride, 1), padding=((2, 2), (0, 0)), rngs=rngs),
            nnx.Conv(128, 512, kernel_size=(kernel_size, 1), strides=(stride, 1), padding=((2, 2), (0, 0)), rngs=rngs),
            nnx.Conv(512, 1024, kernel_size=(kernel_size, 1), strides=(stride, 1), padding=((2, 2), (0, 0)), rngs=rngs),
            nnx.Conv(1024, 1024, kernel_size=(kernel_size, 1), strides=(1, 1), padding=((2, 2), (0, 0)), rngs=rngs),
        ]

        self.conv_post = nnx.Conv(1024, 1, kernel_size=(3, 1), strides=(1, 1), padding=((1, 1), (0, 0)), rngs=rngs)

    def __call__(self, x: jax.Array) -> tuple[jax.Array, tuple[jax.Array, ...]]:
        batch_size, time_steps = x.shape

        # Use static padding to avoid dynamic control flow
        pad_size = (self.period - (time_steps % self.period)) % self.period
        x = jnp.pad(x, ((0, 0), (0, pad_size)), mode="constant", constant_values=0)
        padded_time = x.shape[1]

        # Use static reshape with known dimensions
        x = x.reshape(batch_size, padded_time // self.period, self.period)
        x = jnp.expand_dims(x, axis=-1)

        # Pre-allocate tuple for features (JIT-friendly)
        features = []
        for i, conv in enumerate(self.convs):
            x = conv(x)
            x = nnx.leaky_relu(x, negative_slope=0.1)
            features.append(x)

        x = self.conv_post(x)
        features.append(x)

        # Use tuple instead of list for JIT efficiency
        features = tuple(features)

        # Static reshape for output
        x = x.reshape(batch_size, -1, 1)

        return x, features


class OptimizedMultiPeriodDiscriminator(nnx.Module):
    """Memory-efficient MPD with static operations."""

    def __init__(self, periods: tuple[int, ...] = (2, 3, 5, 7, 11), kernel_size: int = 5, stride: int = 3, rngs: nnx.Rngs = None):
        # Use tuple instead of list for static compilation
        self.periods = periods
        self.discriminators = tuple(
            OptimizedPeriodDiscriminator(period, kernel_size, stride, rngs)
            for period in periods
        )

    def __call__(self, x: jax.Array) -> tuple[tuple[jax.Array, ...], tuple[tuple[jax.Array, ...], ...]]:
        if x.ndim == 3 and x.shape[-1] == 1:
            x = x.squeeze(-1)

        # Pre-allocate outputs as tuples
        outputs = []
        all_features = []

        for disc in self.discriminators:
            output, features = disc(x)
            outputs.append(output)
            all_features.append(features)

        return tuple(outputs), tuple(all_features)