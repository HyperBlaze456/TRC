from flax import nnx
from tokenizer.utils.norm import WeightNorm
import jax


class ScaleDiscriminator(nnx.Module):
    """Single scale discriminator that operates on audio at a specific downsampling rate.
    
    Based on DAC's multi-scale discriminator architecture.
    """

    def __init__(
        self,
        rate: int = 1,
        channels: list[int] = [16, 64, 256, 1024, 1024, 1024],
        kernel_size: int = 15,
        groups: list[int] = [1, 4, 16, 64, 256, 1],
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
            s = 1  # make this later on somewhat configurable via parameter `strides`, which can be a list later on?

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

    def __call__(self, x: jax.Array, training: bool = True) -> tuple[jax.Array, list[jax.Array]]:
        """Forward pass returning output and intermediate features.
        
        Args:
            x: Input audio [B, T, 1]
            training: Whether in training mode
            
        Returns:
            output: Discriminator output [B, T', 1]
            features: List of intermediate feature maps
        """
        features = []

        # Downsample input if rate > 1
        if self.rate > 1:
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

        return x, features


class MultiScaleDiscriminator(nnx.Module):
    """Multi-scale discriminator combining discriminators at different temporal resolutions.
    
    Uses 3 discriminators operating on:
    - Original audio (rate=1)
    - 2x downsampled audio (rate=2)  
    - 4x downsampled audio (rate=4)
    """

    def __init__(
        self,
        rates: list[int] = [1, 2, 4],
        channels: list[int] = [16, 64, 256, 1024, 1024, 1024],
        kernel_size: int = 15,
        groups: list[int] = [1, 4, 16, 64, 256, 1],
        rngs: nnx.Rngs = None
    ):
        if rngs is None:
            rngs = nnx.Rngs(0)

        self.discriminators = []
        for rate in rates:
            disc = ScaleDiscriminator(
                rate=rate,
                channels=channels,
                kernel_size=kernel_size,
                groups=groups,
                rngs=rngs
            )
            self.discriminators.append(disc)

    def __call__(self, x: jax.Array, training: bool = True) -> tuple[list[jax.Array], list[list[jax.Array]]]:
        """Forward pass through all scale discriminators.
        
        Args:
            x: Input audio [B, T, 1]
            training: Whether in training mode
            
        Returns:
            outputs: List of discriminator outputs, one per scale
            features: List of feature lists, one per scale
        """
        outputs = []
        all_features = []

        for disc in self.discriminators:
            output, features = disc(x, training=training)
            outputs.append(output)
            all_features.append(features)

        return outputs, all_features

