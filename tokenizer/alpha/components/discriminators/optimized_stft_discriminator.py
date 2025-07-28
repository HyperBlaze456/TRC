from flax import nnx
import jax
import jax.numpy as jnp
from functools import partial


class OptimizedSTFTDiscriminator(nnx.Module):
    """Memory-efficient STFT discriminator with static operations."""

    def __init__(self,
                 fft_sizes: tuple[int, ...] = (2048, 1024, 512),
                 hop_lengths: tuple[int, ...] | None = None,
                 win_lengths: tuple[int, ...] | None = None,
                 channels: tuple[int, ...] = (32, 64, 128, 256, 512),
                 kernel_size: tuple[int, int] = (3, 9),
                 strides: tuple[int, int] = (1, 2),
                 padding: str = "SAME",
                 rngs: nnx.Rngs = None):
        if rngs is None:
            rngs = nnx.Rngs(0)

        # Use tuples for static compilation
        self.fft_sizes = fft_sizes
        self.hop_lengths = hop_lengths or tuple(n // 4 for n in fft_sizes)
        self.win_lengths = win_lengths or fft_sizes

        # Pre-compute windows for each FFT size
        self.windows = tuple(
            self._create_window(win, fft)
            for win, fft in zip(self.win_lengths, self.fft_sizes)
        )

        self.discriminators = tuple(
            STFTResolutionDiscriminator(
                channels=channels,
                kernel_size=kernel_size,
                strides=strides,
                padding=padding,
                rngs=rngs
            )
            for _ in range(len(fft_sizes))
        )

    @staticmethod
    def _create_window(win_length: int, fft_size: int) -> jax.Array:
        """Pre-compute window to avoid dynamic allocation."""
        window = jnp.hanning(win_length)
        if win_length < fft_size:
            pad = (fft_size - win_length) // 2
            window = jnp.pad(window, (pad, fft_size - win_length - pad))
        return window

    @partial(jax.jit, static_argnames=['fft_size', 'hop'])
    def _stft_static(self, x: jax.Array, window: jax.Array, fft_size: int, hop: int) -> jax.Array:
        """Static STFT computation for better JIT compilation."""
        if x.ndim == 3:
            x = x.squeeze(-1)

        batch_size, time_length = x.shape
        
        # Compute number of frames statically
        num_frames = 1 + (time_length - fft_size) // hop
        
        # Use vmap instead of explicit indexing for better memory efficiency
        def extract_frame(start_idx):
            return x[:, start_idx:start_idx + fft_size] * window
        
        # Vectorized frame extraction
        starts = jnp.arange(num_frames) * hop
        frames = jax.vmap(extract_frame, in_axes=0, out_axes=1)(starts)
        
        # FFT computation
        spec = jnp.fft.rfft(frames, n=fft_size, axis=-1)
        return spec

    def __call__(self, x: jax.Array, *, training: bool = True) -> tuple[tuple[jax.Array, ...], tuple[tuple[jax.Array, ...], ...]]:
        outputs = []
        all_features = []

        for i, (fft_size, hop, window) in enumerate(zip(
            self.fft_sizes, self.hop_lengths, self.windows
        )):
            # Static STFT computation
            spec = self._stft_static(x, window, fft_size, hop)
            
            # Stack magnitude and phase
            feat = jnp.stack([jnp.abs(spec), jnp.angle(spec)], -1)
            
            # Apply discriminator
            out, features = self.discriminators[i](feat, training=training)
            outputs.append(out)
            all_features.append(features)

        return tuple(outputs), tuple(all_features)


class STFTResolutionDiscriminator(nnx.Module):
    """Optimized 2D Conv discriminator."""

    def __init__(self,
                 channels: tuple[int, ...],
                 kernel_size: tuple[int, int],
                 strides: tuple[int, int],
                 padding: str,
                 rngs: nnx.Rngs):
        self.convs = []
        c_in = 2

        for c_out in channels:
            conv = nnx.Conv(c_in, c_out,
                            kernel_size=kernel_size,
                            strides=strides,
                            padding=padding,
                            rngs=rngs)
            self.convs.append(conv)
            c_in = c_out

        self.out_conv = nnx.Conv(c_in, 1,
                                 kernel_size=kernel_size,
                                 strides=1,
                                 padding=padding,
                                 rngs=rngs)

    def __call__(self, x: jax.Array, *, training: bool) -> tuple[jax.Array, tuple[jax.Array, ...]]:
        features = []
        for conv in self.convs:
            x = conv(x)
            x = nnx.elu(x)
            features.append(x)

        x = self.out_conv(x)
        features.append(x)

        # Use tuple for static compilation
        features = tuple(features)

        # Global average pooling
        x = jnp.mean(x, axis=(1, 2))
        return x.squeeze(-1), features