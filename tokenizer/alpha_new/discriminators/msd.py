from flax import nnx
import jax
import jax.numpy as jnp

class ScaleDiscriminator(nnx.Module):
    def __init__(self, rate: int, rngs: nnx.Rngs):
        self.rate = rate
        self.convs = [
            nnx.Conv(1, 16, 15, 1, padding=7, rngs=rngs),
            nnx.Conv(16, 64, 41, 4, padding=20, feature_group_count=4, rngs=rngs),
            nnx.Conv(64, 256, 41, 4, padding=20, feature_group_count=16, rngs=rngs),
            nnx.Conv(256, 1024, 41, 4, padding=20, feature_group_count=64, rngs=rngs),
            nnx.Conv(1024, 1024, 41, 4, padding=20, feature_group_count=256, rngs=rngs),
            nnx.Conv(1024, 1024, 5, 1, padding=2, rngs=rngs),
        ]
        self.conv_post = nnx.Conv(1024, 1, 3, 1, padding=1, rngs=rngs)

    def __call__(self, x: jax.Array):
        """
        Args:
            x: Input tensor of shape [B, T, 1]
        Returns:
            feature_map: Contains output features from each convolutional layer
        """
        # Downsample input if rate > 1
        if self.rate > 1:
            x = nnx.avg_pool(
                x, window_shape=(self.rate,), strides=(self.rate,), padding="VALID"
            )

        feature_map = []
        for conv in self.convs:
            x = conv(x)
            x = nnx.leaky_relu(x, negative_slope=0.1)
            feature_map.append(x)

        x = self.conv_post(x)
        feature_map.append(x)

        return feature_map

class MSD(nnx.Module):
    def __init__(
            self,
            rates: list[int] = [1, 2, 4],
            rngs: nnx.Rngs = None,
    ):
        self.discriminators = [
            ScaleDiscriminator(rate, rngs) for rate in rates
        ]

    def __call__(self, x: jax.Array):
        """
        Args:
            x: Input tensor of shape [B, T, 1]
        Returns:
            feature_maps: List of feature maps from each scale discriminator
        """
        feature_maps = []
        for discriminator in self.discriminators:
            feature_map = discriminator(x)
            feature_maps.append(feature_map)

        return feature_maps

if __name__ == '__main__':
    key = jax.random.PRNGKey(42)
    rngs = nnx.Rngs(0)
    discriminator = ScaleDiscriminator(2, rngs)
    audio = jax.random.normal(key, shape=(32, 168_000, 1))

    featmap = discriminator(audio)
    print(len(featmap))
    print(featmap[-1].shape)
