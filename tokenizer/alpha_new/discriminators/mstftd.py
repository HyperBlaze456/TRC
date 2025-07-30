from flax import nnx
import jax
import jax.numpy as jnp
import os

# Enable memory profiling
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.8'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'


class STFTDiscriminator(nnx.Module):
    def __init__(self, fft_size: int, hop_length: int, win_length: int, rngs: nnx.Rngs):
        self.fft_size = fft_size
        self.hop_length = hop_length
        self.win_length = win_length

        self.convs = [
            nnx.Conv(
                2, 32,
                kernel_size=(3, 9),
                strides=(1, 2),
                padding="SAME",
                rngs=rngs
            ),
            nnx.Conv(
                32, 64,
                kernel_size=(3, 9),
                strides=(1, 2),
                padding="SAME",
                rngs=rngs
            ),
            nnx.Conv(
                64, 128,
                kernel_size=(3, 9),
                strides=(1, 2),
                padding="SAME",
                rngs=rngs
            ),
            nnx.Conv(
                128, 256,
                kernel_size=(3, 9),
                strides=(1, 2),
                padding="SAME",
                rngs=rngs
            ),
            nnx.Conv(
                256, 512,
                kernel_size=(3, 9),
                strides=(1, 2),
                padding="SAME",
                rngs=rngs
            ),
        ]

        self.conv_post = nnx.Conv(
            512, 1,
            kernel_size=(3, 9),
            strides=(1, 1),
            padding="SAME",
            rngs=rngs
        )

    def __call__(self, x: jax.Array):
        """
        Args:
            x: Input tensor of shape [B, T, 1]
        Returns:
            feature_map: Contains output features from each convolutional layer
        """

        x = x.squeeze(-1)  # [B, T]

        # Returns (freqs, times, Zxx) where Zxx is complex STFT
        _, _, Zxx = jax.scipy.signal.stft(
            x,
            nperseg=self.win_length,
            noverlap=self.win_length - self.hop_length,
            nfft=self.fft_size,
            boundary=None,
            padded=False
        )

        jax.debug.print("STFT output shape: {shape}", shape=Zxx.shape)

        # Zxx shape: [B, F, T_frames]
        # Stack magnitude and phase as channels
        mag = jnp.abs(Zxx)
        phase = jnp.angle(Zxx)

        # Transpose to [B, T_frames, F, 2] for 2D convolution
        x = jnp.stack([mag, phase], axis=-1)  # [B, F, T_frames, 2]
        x = jnp.transpose(x, (0, 2, 1, 3))  # [B, T_frames, F, 2]

        feature_map = []
        for i, conv in enumerate(self.convs):
            x = conv(x)
            x = nnx.leaky_relu(x, negative_slope=0.1)
            feature_map.append(x)

        x = self.conv_post(x)
        feature_map.append(x)

        return feature_map


class MSTFTD(nnx.Module):
    def __init__(
            self,
            fft_sizes: list[int] = [2048, 1024, 512],
            hop_lengths: list[int] = [512, 256, 128],
            win_lengths: list[int] = [2048, 1024, 512],
            rngs: nnx.Rngs = None,
    ):
        self.discriminators = []
        for fft_size, hop_length, win_length in zip(fft_sizes, hop_lengths, win_lengths):
            disc = STFTDiscriminator(
                fft_size=fft_size,
                hop_length=hop_length,
                win_length=win_length,
                rngs=rngs
            )
            self.discriminators.append(disc)

    def __call__(self, x: jax.Array):
        """
        Args:
            x: Input tensor of shape [B, T, 1]
        Returns:
            feature_maps: List of feature maps from each STFT resolution
        """
        feature_maps = []
        for discriminator in self.discriminators:
            feature_map = discriminator(x)
            feature_maps.append(feature_map)

        return feature_maps


@nnx.jit
def run_inference(model, audio):
    """JIT-compiled inference function"""
    return model(audio)


if __name__ == '__main__':
    jax.config.update('jax_enable_x64', False)

    key = jax.random.PRNGKey(42)
    rngs = nnx.Rngs(0)

    try:
        jax.profiler.start_server(6009)
        mstftd = MSTFTD(rngs=rngs)

        audio = jax.random.normal(key, shape=(32, 168_000, 1))
        print(f"Testing MSTFTD with input shape: {audio.shape}")


        featmaps = run_inference(mstftd, audio)

        jax.block_until_ready(featmaps)

        # Print results
        print(f"Number of resolutions: {len(featmaps)}")

        for i, resolution_featmaps in enumerate(featmaps):
            print(f"\nResolution {i} (fft_size={[2048, 1024, 512][i]}):")
            print(f"  Number of feature maps: {len(resolution_featmaps)}")
            print(f"  Final output shape: {resolution_featmaps[-1].shape}")

    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback

        traceback.print_exc()

