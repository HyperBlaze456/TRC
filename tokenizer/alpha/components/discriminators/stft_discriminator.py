
from flax import nnx
import jax
import jax.numpy as jnp


class STFTDiscriminator(nnx.Module):
    """Multi-resolution STFT discriminator. Proposed from SoundStream, used in DAC too."""

    def __init__(self,
                 fft_sizes: list[int] = (2048, 1024, 512),
                 hop_lengths: list[int] | None = None,
                 win_lengths: list[int] | None = None,
                 channels: list[int] = (32, 64, 128, 256, 512),
                 kernel_size: tuple[int, int] = (3, 9),
                 strides: tuple[int, int] = (1, 2),
                 padding: str = "SAME",
                 rngs: nnx.Rngs = None):
        if rngs is None:
            rngs = nnx.Rngs(0)

        self.fft_sizes = list(fft_sizes)
        self.hop_lengths = hop_lengths or [n // 4 for n in self.fft_sizes]
        self.win_lengths = win_lengths or self.fft_sizes

        self.discriminators = []
        for _ in range(len(self.fft_sizes)):
            disc = STFTResolutionDiscriminator(
                channels=channels,
                kernel_size=kernel_size,
                strides=strides,
                padding=padding,
                rngs=rngs
            )
            self.discriminators.append(disc)

    def _stft(self,
              x: jax.Array,
              fft_size: int,
              hop: int,
              win: int) -> jax.Array:
        """Return complex STFT with shape [B, F, T_f]."""
        if x.ndim == 3:
            x = jnp.squeeze(x, -1)  # [B, T]

        window = jnp.hanning(win).astype(x.dtype)
        if win < fft_size:
            pad = (fft_size - win) // 2
            window = jnp.pad(window, (pad, pad))

        num_frames = 1 + (x.shape[1] - fft_size) // hop
        starts = jnp.arange(num_frames) * hop
        idx = starts[:, None] + jnp.arange(fft_size)
        frames = x[:, idx] * window

        spec = jnp.fft.rfft(frames, n=fft_size, axis=-1)
        return spec

    def _pad_to_max(self, feats: list[jax.Array]) -> jax.Array:
        """Zero‑pad each [B, 2, F, T] to common size and stack on axis‑0."""
        f_max = max(t.shape[2] for t in feats)
        t_max = max(t.shape[3] for t in feats)
        padded = [jnp.pad(f,
                          ((0, 0), (0, 0),
                           (0, f_max - f.shape[2]),
                           (0, t_max - f.shape[3])))
                  for f in feats]
        return jnp.stack(padded, 0)

    def __call__(self, x: jax.Array, *, training: bool = True) -> tuple[list[jax.Array], list[list[jax.Array]]]:
        outputs = []
        all_features = []

        for i, (fft, hop, win) in enumerate(zip(self.fft_sizes,
                                                self.hop_lengths,
                                                self.win_lengths, strict=False)):
            # Compute STFT
            spec = self._stft(x, fft, hop, win)

            # Stack magnitude and phase
            feat = jnp.stack([jnp.abs(spec), jnp.angle(spec)], -1)  # [B, T_f, F, 2]

            # Apply discriminator
            out, features = self.discriminators[i](feat, training=training)
            outputs.append(out)
            all_features.append(features)

        return outputs, all_features


class STFTResolutionDiscriminator(nnx.Module):
    """2D Conv discriminator for a single STFT resolution (no batch norm)."""

    def __init__(self,
                 channels: list[int],
                 kernel_size: tuple[int, int],
                 strides: tuple[int, int],
                 padding: str,
                 rngs: nnx.Rngs):
        self.convs = []
        c_in = 2  # magnitude and phase channels

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

    def __call__(self, x: jax.Array, *, training: bool) -> tuple[jax.Array, list[jax.Array]]:
        # Input must be in the shape of [B, T_f, F, 2]. That 2 is the channel.
        # Apply conv layers with ELU activation
        features = []
        for conv in self.convs:
            x = conv(x)
            x = nnx.elu(x)
            features.append(x)

        # Final conv to single channel
        x = self.out_conv(x)
        features.append(x)

        # Global average pooling
        x = jnp.mean(x, axis=(1, 2))
        return jnp.squeeze(x, -1), features  # [B], List of features
