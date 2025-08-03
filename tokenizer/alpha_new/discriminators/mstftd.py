from flax import nnx
import jax
import jax.numpy as jnp

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

# Can't JIT print

if __name__ == '__main__':
    # Get initial memory state
    devices = jax.devices()
    print(f"Running on devices: {devices}")
    
    if devices and devices[0].platform == "gpu":
        initial_stats = devices[0].memory_stats()
        if initial_stats:
            print(f"Initial GPU memory: {initial_stats.get('bytes_in_use', 0) / (1024**3):.3f} GB")
    
    key = jax.random.PRNGKey(42)
    rngs = nnx.Rngs(0)

    model = MSTFTD(rngs=rngs)
    mock_audio = jax.random.normal(key, (4, 168_000, 1))

    # Run without JIT to see memory usage properly
    print("\nRunning MSTFTD forward pass (no JIT)...")
    featmap = model(mock_audio)
    
    # Ensure all computations are complete
    for scale_maps in featmap:
        for feat in scale_maps:
            jax.block_until_ready(feat)
    
    # Get final memory state
    if devices and devices[0].platform == "gpu":
        final_stats = devices[0].memory_stats()
        if final_stats:
            print(f"\nFinal GPU memory: {final_stats.get('bytes_in_use', 0) / (1024**3):.3f} GB")
            print(f"Total memory allocated: {final_stats.get('bytes_in_use', 0) / (1024**3):.3f} GB")
            
            if initial_stats:
                delta = (final_stats.get('bytes_in_use', 0) - initial_stats.get('bytes_in_use', 0)) / (1024**3)
                print(f"Memory increase: {delta:.3f} GB")