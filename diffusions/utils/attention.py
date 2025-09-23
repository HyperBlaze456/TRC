import jax
from flax import nnx
from .embeddings import RotaryPositionalEmbedding

class MHA(nnx.Module):
    def __init__(
            self,
            num_heads: int,
            in_features: int,
            rngs: nnx.Rngs,
            qkv_features: int = None,
            out_features: int = None,
            dropout_rate: float = 0.0,
            use_bias: bool = True,
            rope: RotaryPositionalEmbedding | None = None,
    ):
        self.num_heads = num_heads
        self.in_features = in_features
        self.qkv_features = qkv_features or in_features
        self.out_features = out_features or in_features
        self.dropout_rate = dropout_rate
        self.rngs = rngs
        if rope is not None:
            self.rope = rope
        else:
            self.rope = RotaryPositionalEmbedding(self.qkv_features)

        if self.qkv_features % self.num_heads != 0:
            raise ValueError(
                f"qkv_features ({self.qkv_features}) must be divisible by "
                f"num_heads ({self.num_heads})"
            )

        self.head_dim = self.qkv_features // self.num_heads

        self.q_proj = nnx.Linear(in_features, self.qkv_features, use_bias=use_bias, rngs=rngs)
        self.k_proj = nnx.Linear(in_features, self.qkv_features, use_bias=use_bias, rngs=rngs)
        self.v_proj = nnx.Linear(in_features, self.qkv_features, use_bias=use_bias, rngs=rngs)
        self.out_proj = nnx.Linear(self.qkv_features, self.out_features, use_bias=use_bias, rngs=rngs)

        if dropout_rate > 0.0:
            self.dropout = nnx.Dropout(dropout_rate, rngs=rngs)
        else:
            self.dropout = None

    def __call__(
            self,
            query: jax.Array,
            key: jax.Array | None = None,
            value: jax.Array | None = None,
            mask: jax.Array | None = None,
            deterministic: bool = False,
    ) -> jax.Array:
        if key is None:
            key = query
        if value is None:
            value = query

        batch_size, seq_len = query.shape[:2]

        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        if self.rope is not None:
            cos, sin = self.rope(seq_len, dtype=q.dtype)
            q = self.rope.apply_rotary_pos_emb(q, cos, sin)
            k = self.rope.apply_rotary_pos_emb(k, cos, sin)

        q = q.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v.reshape(batch_size, seq_len, self.num_heads, self.head_dim)

        if mask is not None:
            if mask.ndim == 3:
                mask = mask[:, None, :, :]
            elif mask.ndim == 2:
                mask = mask[None, None, :, :]

        dropout_rng = None
        if self.dropout_rate > 0.0 and not deterministic:
            dropout_rng = self.rngs.dropout()

        attn_output = nnx.dot_product_attention(
            q,
            k,
            v,
            mask=mask,
            dropout_rate=self.dropout_rate if not deterministic else 0.0,
            dropout_rng=dropout_rng,
        )

        attn_output = attn_output.reshape(batch_size, seq_len, self.qkv_features)

        output = self.out_proj(attn_output)

        if self.dropout is not None and not deterministic:
            output = self.dropout(output)

        return output


