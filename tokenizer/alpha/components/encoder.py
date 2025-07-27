from flax import nnx

from tokenizer.utils.activation import Snake
from tokenizer.utils.attention import (
    MultiHeadAttentionWithRoPE,
    RotaryPositionalEmbedding,
)


class DepthwiseUnit(nnx.Module):
    """Depthwise separable convolution: Depthwise Conv followed by Pointwise Conv.
    """
    def __init__(self, in_features, out_features, kernel_size, stride:int, rngs:nnx.Rngs, padding_mode: str = "CAUSAL"):
        self.conv_depth = nnx.Conv(
            in_features=in_features,
            out_features=in_features,
            kernel_size=kernel_size,
            strides=stride,
            feature_group_count=in_features,
            padding=padding_mode,
            rngs=rngs
        )
        self.conv_point = nnx.Conv(
            in_features=in_features,
            out_features=out_features,
            kernel_size=1,
            strides=1,
            padding="VALID",
            rngs=rngs
        )
        self.snake = Snake(out_features)

    def __call__(self, x):
        x = self.conv_depth(x)
        x = self.snake(x)
        x = self.conv_point(x)
        x = self.snake(x)
        return x

class TransformerBlock(nnx.Module):
    """Transformer block with attention, MLP, residual connections, and Snake activation."""
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float,
        rngs: nnx.Rngs,
        dropout_rate: float = 0.0,
        rope: RotaryPositionalEmbedding = None,
    ):
        self.attention = MultiHeadAttentionWithRoPE(
            in_features=hidden_size,
            num_heads=num_heads,
            rngs=rngs,
            dropout_rate=dropout_rate,
            rope=rope
        )

        # MLP with expansion
        mlp_hidden_size = int(hidden_size * mlp_ratio)
        self.mlp = nnx.Sequential(
            nnx.Linear(hidden_size, mlp_hidden_size, rngs=rngs),
            Snake(mlp_hidden_size),
            nnx.Linear(mlp_hidden_size, hidden_size, rngs=rngs),
        )

        # Layer norms
        self.norm1 = nnx.RMSNorm(hidden_size, rngs=rngs)
        self.norm2 = nnx.RMSNorm(hidden_size, rngs=rngs)

    def __call__(self, x, mask=None, rope_cos_sin=None):
        # Attention block with residual
        attn_out = self.attention(
            self.norm1(x),
            mask=mask,
            rope_cos_sin=rope_cos_sin
        )
        x = x + attn_out

        # MLP block with residual
        mlp_out = self.mlp(self.norm2(x))
        x = x + mlp_out

        return x


class RawEncoder(nnx.Module):
    """Compresses raw waveform time-wise, while reducing bps too.
    Allows 48khz/24khz input of shape [B, N, 1], compressing it down to 50hz time-wise.
    After compressing it down, it will be passed through self-attention using MHA, RoPE for positional embeddings.
    """
    def __init__(
            self,
            hidden_size: int,
            depth: int,
            num_heads: int,
            rngs: nnx.Rngs,
            mlp_ratio: float = 4.0,
            is_48khz: bool = False,
    ):
        self.is_48khz = is_48khz
        if self.is_48khz:
            self.conv_48to24 = nnx.Conv(
                in_features=1,
                out_features=1,
                kernel_size=4,
                strides=2,
                padding="CAUSAL",
                rngs=rngs,
            )

        # 24kHz -> 50Hz = 480x compression

        # Stage 1: 24kHz -> 6kHz (stride 4)
        self.conv1 = nnx.Conv(
            in_features=1,
            out_features=8,
            kernel_size=12,
            strides=4,
            padding="CAUSAL",
            rngs=rngs,
        )
        self.snake1 = Snake(8)

        # Stage 2: 6kHz -> 2kHz (stride 3)
        self.conv2 = nnx.Conv(
            in_features=8,
            out_features=16,
            kernel_size=5,
            strides=3,
            padding="CAUSAL",
            rngs=rngs,
        )
        self.snake2 = Snake(16)

        # Stage 3: 2kHz -> 400Hz (stride 5)
        self.conv3 = nnx.Conv(
            in_features=16,
            out_features=64,
            kernel_size=9,
            strides=5,
            padding="CAUSAL",
            rngs=rngs,
        )
        self.snake3 = Snake(64)

        # Stage 4: 400Hz -> 50Hz (stride 8), this can be
        # recommended hidden size to be at least 80, optimally 128 or more...
        self.conv4 = nnx.Conv(
            in_features=64,
            out_features=hidden_size,
            kernel_size=12,
            strides=8,
            padding="CAUSAL",
            rngs=rngs,
        )
        self.snake4 = Snake(hidden_size)

        # RoPE positional embedding
        self.rope = RotaryPositionalEmbedding(hidden_size // num_heads)

        # Stack of transformer blocks
        self.transformer_blocks = [
            TransformerBlock(
                hidden_size=hidden_size,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                rngs=rngs,
                rope=self.rope
            )
            for _ in range(depth)
        ]

    def __call__(self, x, mask=None):
        # x: [B, T, 1] where T is at 48kHz or 24kHz

        # Optional 48kHz to 24kHz conversion
        if self.is_48khz:
            x = self.conv_48to24(x)

        # Progressive downsampling
        x = self.conv1(x)
        x = self.snake1(x)

        x = self.conv2(x)
        x = self.snake2(x)

        x = self.conv3(x)
        x = self.snake3(x)

        x = self.conv4(x)
        x = self.snake4(x)

        # Now x is [B, T/480, hidden_size] at 50Hz
        seq_len = x.shape[1]

        # Generate RoPE embeddings once for all layers
        rope_cos_sin = self.rope(seq_len, dtype=x.dtype)

        # Apply transformer blocks
        for block in self.transformer_blocks:
            x = block(x, mask=mask, rope_cos_sin=rope_cos_sin)

        return x
