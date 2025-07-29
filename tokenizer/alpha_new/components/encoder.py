from flax import nnx

from tokenizer.utils.activation import Snake
from tokenizer.utils.attention import (
    MultiHeadAttentionWithRoPE,
    RotaryPositionalEmbedding,
)


class TransformerBlock(nnx.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_dim: int,
        rngs: nnx.Rngs,
        dropout_rate: float = 0.1,
        rope: RotaryPositionalEmbedding = None,
    ):
        self.attention = MultiHeadAttentionWithRoPE(
            in_features=dim,
            num_heads=num_heads,
            rngs=rngs,
            dropout_rate=dropout_rate,
            rope=rope,
        )
        self.mlp = nnx.Sequential(
            nnx.Linear(dim, mlp_dim, rngs=rngs),
            Snake(mlp_dim),
            nnx.Linear(mlp_dim, dim, rngs=rngs),
        )

        self.norm1 = nnx.RMSNorm(dim, rngs=rngs)
        self.norm2 = nnx.RMSNorm(dim, rngs=rngs)

    def __call__(self, x, mask=None, rope_cos_sin=None):
        attn_out = self.attention(self.norm1(x), mask=mask, rope_cos_sin=rope_cos_sin)
        x = x + attn_out

        # MLP block with residual
        mlp_out = self.mlp(self.norm2(x))
        x = x + mlp_out

        return x


class RawEncoder(nnx.Module):
    def __init__(
        self,
        hidden_size: int,
        depth: int,
        num_heads: int,
        rngs: nnx.Rngs,
        mlp_dim: int,
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

        self.conv1 = nnx.Conv(
            in_features=1,
            out_features=8,
            kernel_size=12,
            strides=4,
            padding="CAUSAL",
            rngs=rngs,
        )
        self.snake1 = Snake(8)

        self.conv2 = nnx.Conv(
            in_features=8,
            out_features=16,
            kernel_size=5,
            strides=3,
            padding="CAUSAL",
            rngs=rngs,
        )
        self.snake2 = Snake(16)

        self.conv3 = nnx.Conv(
            in_features=16,
            out_features=64,
            kernel_size=9,
            strides=5,
            padding="CAUSAL",
            rngs=rngs,
        )
        self.snake3 = Snake(64)

        self.conv4 = nnx.Conv(
            in_features=64,
            out_features=hidden_size,
            kernel_size=12,
            strides=8,
            padding="CAUSAL",
            rngs=rngs,
        )
        self.snake4 = Snake(hidden_size)

        self.rope = RotaryPositionalEmbedding(hidden_size // num_heads)

        self.transformer_blocks = [
            TransformerBlock(
                dim=hidden_size,
                num_heads=num_heads,
                mlp_dim=mlp_dim if mlp_dim > 0 else hidden_size * 4,
                rngs=rngs,
                rope=self.rope,
            )
            for _ in range(depth)
        ]

    def __call__(self, x, mask=None):
        if self.is_48khz:
            x = self.conv_48to24(x)

        x = self.conv1(x)
        x = self.snake1(x)

        x = self.conv2(x)
        x = self.snake2(x)

        x = self.conv3(x)
        x = self.snake3(x)

        x = self.conv4(x)
        x = self.snake4(x)

        seq_len = x.shape[1]
        rope_cos_sin = self.rope(seq_len)

        for block in self.transformer_blocks:
            x = block(x, mask=mask, rope_cos_sin=rope_cos_sin)

        return x

    def encode(self):
        """TODO"""
