from flax import nnx
import jax
import jax.numpy as jnp


class OptimizedScaleDiscriminator(nnx.Module):
    """Memory-efficient single scale discriminator with JIT-friendly operations."""

    def __init__(
        self,
        rate: int = 1,
        channels: tuple[int, ...] = (16, 64, 256, 1024, 1024, 1024),
        kernel_size: int = 15,
        groups: tuple[int, ...] = (1, 4, 16, 64, 256, 1),
        strides: int = 1,
        rngs: nnx.Rngs = None
    ):
        if rngs is None:
            rngs = nnx.Rngs(0)

        self.rate = rate

        # Build conv layers
        self.convs = []
        self.norms = []

        in_channels = 1  # Mono audio input
        for i, (out_channels, group) in enumerate(zip(channels, groups, strict=False)):
            # Use final kernel size and stride for last layer
            is_last = i == len(channels) - 1
            k = 3 if is_last else kernel_size
            s = 1  # Fixed stride

            conv = nnx.Conv(
                in_features=in_channels,
                out_features=out_channels,
                kernel_size=k,
                strides=s,
                padding="SAME",
                feature_group_count=group,
                rngs=rngs
            )
            self.convs.append(conv)

            # Add group norm for all layers except last
            if not is_last:
                norm = nnx.GroupNorm(
                    num_groups=group,
                    num_features=out_channels,
                    rngs=rngs
                )
                self.norms.append(norm)

            in_channels = out_channels

    def __call__(self, x: jax.Array, training: bool = True) -> tuple[jax.Array, tuple[jax.Array, ...]]:
        """Forward pass returning output and intermediate features."""
        features = []

        # Downsample input if rate > 1 - use static shape operations
        if self.rate > 1:
            # Ensure input is padded to be divisible by rate
            _, time_steps, _ = x.shape
            pad_size = (self.rate - (time_steps % self.rate)) % self.rate
            if pad_size > 0:
                x = jnp.pad(x, ((0, 0), (0, pad_size), (0, 0)), mode="constant")
            
            # Average pooling for downsampling
            x = nnx.avg_pool(x, window_shape=(self.rate,), strides=(self.rate,), padding="VALID")

        # Apply conv layers
        for i, conv in enumerate(self.convs):
            x = conv(x)

            # Apply norm and activation for all but last layer
            if i < len(self.convs) - 1:
                x = self.norms[i](x)
                x = nnx.leaky_relu(x, negative_slope=0.2)
                features.append(x)

        # Return tuple instead of list for JIT efficiency
        return x, tuple(features)


class OptimizedMultiScaleDiscriminator(nnx.Module):
    """Memory-efficient multi-scale discriminator with static operations."""

    def __init__(
        self,
        rates: tuple[int, ...] = (1, 2, 4),
        channels: tuple[int, ...] = (16, 64, 256, 1024, 1024, 1024),
        kernel_size: int = 15,
        groups: tuple[int, ...] = (1, 4, 16, 64, 256, 1),
        rngs: nnx.Rngs = None
    ):
        if rngs is None:
            rngs = nnx.Rngs(0)

        # Use tuples for static compilation
        self.discriminators = tuple(
            OptimizedScaleDiscriminator(
                rate=rate,
                channels=channels,
                kernel_size=kernel_size,
                groups=groups,
                rngs=rngs
            )
            for rate in rates
        )

    def __call__(self, x: jax.Array, training: bool = True) -> tuple[tuple[jax.Array, ...], tuple[tuple[jax.Array, ...], ...]]:
        """Forward pass through all scale discriminators."""
        outputs = []
        all_features = []

        for disc in self.discriminators:
            output, features = disc(x, training=training)
            outputs.append(output)
            all_features.append(features)

        # Return tuples for static compilation
        return tuple(outputs), tuple(all_features)