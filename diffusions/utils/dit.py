"""Diffusion Transformer building blocks.

Implements a DiT-style architecture that conditions attention and MLP blocks
on timestep embeddings (and optional additional conditioning) using the
AdaRMSZero normalization layer.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx

from .attention import MHA
from .embeddings import RotaryPositionalEmbedding, TimeStepEmbedder
from .norm import AdaRMSZero


class DiTBlock(nnx.Module):
    """Single transformer block with AdaRMSZero gating."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        rngs: nnx.Rngs,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        rope: RotaryPositionalEmbedding | None = None,
    ):
        if hidden_size % num_heads != 0:
            raise ValueError(
                "hidden_size must be divisible by num_heads for attention heads"
            )

        self.hidden_size = hidden_size
        self.dropout_rate = dropout

        self.norm1 = AdaRMSZero(hidden_size, rngs=rngs)
        self.attn = MHA(
            num_heads=num_heads,
            in_features=hidden_size,
            qkv_features=hidden_size,
            out_features=hidden_size,
            dropout_rate=dropout,
            rngs=rngs,
            rope=rope,
        )

        self.norm2 = AdaRMSZero(hidden_size, rngs=rngs)
        mlp_hidden = int(hidden_size * mlp_ratio)
        self.mlp = nnx.Sequential(
            nnx.Linear(hidden_size, mlp_hidden, rngs=rngs),
            nnx.gelu,
            nnx.Linear(mlp_hidden, hidden_size, rngs=rngs),
        )

        if dropout > 0.0:
            self.mlp_dropout = nnx.Dropout(dropout, rngs=rngs)
        else:
            self.mlp_dropout = None

    def __call__(
        self,
        x: jax.Array,
        cond: jax.Array,
        mask: jax.Array | None = None,
        deterministic: bool = False,
    ) -> jax.Array:
        residual = x
        h, gate_msa = self.norm1(x, cond)
        attn_out = self.attn(h, mask=mask, deterministic=deterministic)
        x = residual + gate_msa * attn_out

        residual = x
        h, gate_mlp = self.norm2(x, cond)
        mlp_out = self.mlp(h)
        if self.mlp_dropout is not None and not deterministic:
            mlp_out = self.mlp_dropout(mlp_out)
        x = residual + gate_mlp * mlp_out
        return x


class DiT(nnx.Module):
    """Diffusion Transformer backbone.

    Args:
        in_features: Dimensionality of the input tokens.
        hidden_size: Transformer hidden size.
        depth: Number of DiT blocks.
        num_heads: Attention heads per block.
        rngs: Flax NNX RNG container.
        mlp_ratio: Expansion factor for the MLP hidden dimension.
        dropout: Dropout probability applied to attention and MLP outputs.
        conditioning_dim: Optional dimensionality for global conditioning vectors.
        use_pos_emb: Enable rotary positional embeddings in attention blocks.
    """

    def __init__(
        self,
        in_features: int,
        hidden_size: int,
        depth: int,
        num_heads: int,
        rngs: nnx.Rngs,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        conditioning_dim: int | None = None,
        use_pos_emb: bool = True,
    ):
        if hidden_size % num_heads != 0:
            raise ValueError(
                "hidden_size must be divisible by num_heads for attention heads"
            )

        self.in_features = in_features
        self.hidden_size = hidden_size
        self.depth = depth
        self.use_pos_emb = use_pos_emb

        self.input_proj = nnx.Linear(in_features, hidden_size, rngs=rngs)
        self.final_norm = nnx.RMSNorm(hidden_size, rngs=rngs)
        self.output_proj = nnx.Linear(hidden_size, in_features, rngs=rngs)

        self.t_embedder = TimeStepEmbedder(dim=hidden_size, hidden_size=hidden_size, rngs=rngs)

        if conditioning_dim is not None:
            self.cond_proj = nnx.Sequential(
                nnx.Linear(conditioning_dim, hidden_size, rngs=rngs),
                nnx.silu,
                nnx.Linear(hidden_size, hidden_size, rngs=rngs),
            )
        else:
            self.cond_proj = None

        if use_pos_emb:
            self.rope = RotaryPositionalEmbedding(hidden_size)
        else:
            self.rope = None

        self.blocks = [
            DiTBlock(
                hidden_size=hidden_size,
                num_heads=num_heads,
                rngs=rngs,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                rope=self.rope,
            )
            for _ in range(depth)
        ]

    def _prepare_condition(self, temb: jax.Array, conditioning: jax.Array | None) -> jax.Array:
        """Combine timestep embedding with optional conditioning."""

        if conditioning is None:
            return temb

        if self.cond_proj is None:
            raise ValueError(
                "conditioning was provided but conditioning_dim was not set during initialization"
            )

        if conditioning.ndim == 1:
            conditioning = conditioning[None, :]

        if conditioning.ndim != 2:
            raise ValueError(
                "conditioning must be a 1D or 2D array with shape [batch, cond_dim]"
            )

        if conditioning.shape[0] != temb.shape[0]:
            raise ValueError("conditioning batch size must match input batch size")

        cond_emb = self.cond_proj(conditioning)
        cond_emb = cond_emb.astype(temb.dtype)
        return temb + cond_emb

    def __call__(
        self,
        x: jax.Array,
        timesteps: jax.Array,
        conditioning: jax.Array | None = None,
        mask: jax.Array | None = None,
        deterministic: bool = False,
    ) -> jax.Array:
        if x.ndim != 3:
            raise ValueError("Input x must have shape [batch, seq_len, in_features]")

        batch = x.shape[0]

        h = self.input_proj(x)

        if timesteps.ndim == 2 and timesteps.shape[-1] == 1:
            timesteps = jnp.squeeze(timesteps, axis=-1)
        if timesteps.ndim != 1:
            raise ValueError("timesteps must be a 1D array of shape [batch]")
        if timesteps.shape[0] != batch:
            raise ValueError("timesteps batch size must match input batch size")

        temb = self.t_embedder(timesteps)
        cond = self._prepare_condition(temb, conditioning)

        for block in self.blocks:
            h = block(h, cond, mask=mask, deterministic=deterministic)

        h = self.final_norm(h)
        return self.output_proj(h)
