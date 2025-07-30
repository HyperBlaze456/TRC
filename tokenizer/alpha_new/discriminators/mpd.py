from flax import nnx
import jax
import jax.numpy as jnp

class PeriodDiscriminator(nnx.Module):
    def __init__(self, period, rngs: nnx.Rngs):
        self.period = period
        self.convs = [
            nnx.Conv(
                1, 32,
                kernel_size=(5, 1),
                strides=(3, 1),
                padding=(2, 0),
                rngs=rngs
            ),
            nnx.Conv(
                32, 128,
                kernel_size=(5, 1),
                strides=(3, 1),
                padding=(2, 0),
                rngs=rngs
            ),
            nnx.Conv(
                128, 512,
                kernel_size=(5, 1),
                strides=(3, 1),
                padding=(2, 0),
                rngs=rngs
            ),
            nnx.Conv(
                512, 1024,
                kernel_size=(5, 1),
                strides=(3, 1),
                padding=(2, 0),
                rngs=rngs
            ),
            nnx.Conv(
                1024, 1024,
                kernel_size=(5, 1),
                strides=(3, 1),
                padding=(2, 0),
                rngs=rngs
            ),
        ]
        self.conv_post = nnx.Conv(
            1024, 1,
            kernel_size=(3, 1),
            strides=(1, 1),
            padding=(1, 0),
            rngs=rngs
        )
    def pad_to_period(self, x):
        """
        Args:
            x: Input tensor of shape [B, T, 1]
        Returns:
            x: Padded tensor.
        """
        t = x.shape[1]
        # left-pad, constant로 나중에 실험 ㄱ
        x = jnp.pad(x, ((0,0), (0, self.period - t % self.period), (0,0)), mode='reflect')
        return x
    def __call__(self, x: jax.Array):
        """
        Args:
            x: Input tensor of shape [B, T, 1]
        Returns:
            feature_map: Contains output features from each convolutional layer

        """
        x = self.pad_to_period(x)
        B, T, _ = x.shape
        x = x.reshape(B, T // self.period, self.period)

        x = jnp.expand_dims(x, axis=-1) # [B, H, W, C]

        feature_map = []
        for conv in self.convs:
            x = conv(x)
            x = nnx.leaky_relu(x, negative_slope=0.1)
            feature_map.append(x)

        x = self.conv_post(x)
        feature_map.append(x)

        return feature_map

class MPD(nnx.Module):
    def __init__(
            self,
            periods: list[int] = [2, 3, 5, 7, 11],
            rngs: nnx.Rngs = None,
    ):
        self.discriminators = [
            PeriodDiscriminator(period, rngs) for period in periods
        ]
    def __call__(self, x: jax.Array):
        """
        Args:
            x: Input tensor of shape [B, T, 1]
        Returns:
            feature_maps: List of feature maps from each sub-discriminator
        """
        feature_maps = []
        for discriminator in self.discriminators:
            feature_map = discriminator(x)
            feature_maps.append(feature_map)

        return feature_maps

if __name__ == '__main__':
    key = jax.random.PRNGKey(42)
    rngs = nnx.Rngs(0)
    discriminator = PeriodDiscriminator(11, rngs)
    audio = jax.random.normal(key, shape=(32, 168_000, 1))

    featmap = discriminator(audio)
    print(len(featmap))
    print(featmap[-1].shape)
