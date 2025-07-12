import jax.numpy as jnp
from flax import nnx
from typing import List, Tuple, Optional

# -----------------------------------------------------
# Helper functions
# -----------------------------------------------------

_state_axes = nnx.StateAxes({nnx.Param: 0, nnx.BatchStat: 0})

def _build_disc_stack(rngs: nnx.Rngs,
                      num_resolutions: int,
                      channels: List[int],
                      kernel_size: Tuple[int, int],
                      strides: Tuple[int, int],
                      padding: str):
    """Create a stack of independent discriminators along axis‑0."""

    @nnx.split_rngs(splits=num_resolutions)
    @nnx.vmap  # axis‑0
    def _factory(r):
        return STFTResolutionDiscriminator(
            channels=channels,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            rngs=r,
        )

    return _factory(rngs)  # → [R, …]


def _apply_disc_stack(disc_stack, feats, *, training: bool):
    """Vectorised discriminator forward pass (no kwarg leakage to vmap)."""

    @nnx.vmap(in_axes=(_state_axes, 0), out_axes=0)
    def _apply(disc, feat):  # (disc, feat) slice
        return disc(feat, training=training)

    return _apply(disc_stack, feats)  # [R, B, 1]

# -----------------------------------------------------
# Core modules
# -----------------------------------------------------

class STFTDiscriminator(nnx.Module):
    """Multi‑resolution STFT discriminator with vmap‑parallel branches."""

    def __init__(self,
                 fft_sizes: List[int] = (2048, 1024, 512),
                 hop_lengths: Optional[List[int]] = None,
                 win_lengths: Optional[List[int]] = None,
                 channels: List[int] = (32, 64, 128, 256, 512),
                 kernel_size: Tuple[int, int] = (3, 9),
                 strides: Tuple[int, int] = (1, 2),
                 padding: str = "SAME",
                 rngs: nnx.Rngs = None):
        if rngs is None:
            rngs = nnx.Rngs(0)

        self.fft_sizes   = list(fft_sizes)
        self.hop_lengths = hop_lengths or [n // 4 for n in self.fft_sizes]
        self.win_lengths = win_lengths or self.fft_sizes
        self.num_res     = len(self.fft_sizes)

        self._disc_stack = _build_disc_stack(
            rngs,
            self.num_res,
            channels,
            kernel_size,
            strides,
            padding,
        )

    # -------------------------------------------------
    # STFT utilities
    # -------------------------------------------------

    def _stft(self,
              x: jnp.ndarray,
              fft_size: int,
              hop: int,
              win: int) -> jnp.ndarray:
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
        return jnp.transpose(spec, (0, 2, 1))

    def _pad_to_max(self, feats: List[jnp.ndarray]) -> jnp.ndarray:
        """Zero‑pad each [B, 2, F, T] to common size and stack on axis‑0."""
        f_max = max(t.shape[2] for t in feats)
        t_max = max(t.shape[3] for t in feats)
        padded = [jnp.pad(f,
                          ((0, 0), (0, 0),
                           (0, f_max - f.shape[2]),
                           (0, t_max - f.shape[3])))
                  for f in feats]
        return jnp.stack(padded, 0)  # [R, B, 2, F*, T*]

    # -------------------------------------------------
    # Forward pass
    # -------------------------------------------------

    def __call__(self, x: jnp.ndarray, *, training: bool = True):
        feats = []
        for fft, hop, win in zip(self.fft_sizes,
                                 self.hop_lengths,
                                 self.win_lengths):
            spec = self._stft(x, fft, hop, win)
            feats.append(jnp.stack([jnp.abs(spec), jnp.angle(spec)], 1))

        feats = self._pad_to_max(feats)                  # [R, B, 2, F*, T*]
        outs  = _apply_disc_stack(self._disc_stack,
                                  feats,
                                  training=training)   # [R, B, 1]
        return jnp.squeeze(outs, -1)                     # [R, B]


class STFTResolutionDiscriminator(nnx.Module):
    """2‑D Conv discriminator for a single STFT resolution."""

    def __init__(self,
                 channels: List[int],
                 kernel_size: Tuple[int, int],
                 strides: Tuple[int, int],
                 padding: str,
                 rngs: nnx.Rngs):
        self.blocks = []
        c_in = 2
        for i, c_out in enumerate(channels):
            self.blocks.append({
                "conv": nnx.Conv(c_in, c_out,
                                 kernel_size=kernel_size,
                                 strides=strides,
                                 padding=padding,
                                 rngs=rngs),
                "bn":   None if i == 0 else nnx.BatchNorm(c_out, rngs=rngs),
            })
            c_in = c_out

        self.out_conv = nnx.Conv(c_in, 1,
                                 kernel_size=kernel_size,
                                 strides=1,
                                 padding=padding,
                                 rngs=rngs)

    def __call__(self, x: jnp.ndarray, *, training: bool) -> jnp.ndarray:
        x = jnp.transpose(x, (0, 2, 3, 1))
        for blk in self.blocks:
            x = blk["conv"](x)
            if blk["bn"] is not None:
                x = blk["bn"](x, use_running_average=not training)
            x = nnx.elu(x)

        x = self.out_conv(x)
        x = jnp.mean(x, axis=(1, 2))
        return x
