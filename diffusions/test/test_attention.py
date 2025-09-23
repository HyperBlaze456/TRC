"""Unit tests for rotary positional embeddings and attention."""

import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
pytest.importorskip("flax")

from flax import nnx

from diffusions.utils.embeddings import RotaryPositionalEmbedding
from diffusions.utils.attention import MHA


def test_rotary_positional_embedding_preserves_vector_norm():
    dim = 8
    seq_len = 6
    rope = RotaryPositionalEmbedding(dim)

    cos, sin = rope(seq_len)

    assert cos.shape == (seq_len, dim // 2)
    assert sin.shape == (seq_len, dim // 2)

    key = jax.random.key(0)
    x = jax.random.normal(key, (2, seq_len, dim))

    rotated = rope.apply_rotary_pos_emb(x, cos, sin)

    assert rotated.shape == x.shape

    original_norm = jnp.linalg.norm(x, axis=-1)
    rotated_norm = jnp.linalg.norm(rotated, axis=-1)
    assert jnp.allclose(rotated_norm, original_norm, atol=1e-6)


def test_mha_forward_with_rope_runs_and_is_deterministic():
    rng_key = jax.random.key(123)
    rngs = nnx.Rngs(params=rng_key)

    mha = MHA(
        num_heads=2,
        in_features=8,
        qkv_features=8,
        rngs=rngs,
        dropout_rate=0.0,
    )

    x_key = jax.random.key(7)
    query = jax.random.normal(x_key, (2, 5, 8))

    output = mha(query, deterministic=True)

    assert output.shape == (2, 5, 8)
    assert jnp.isfinite(output).all()

    repeat = mha(query, deterministic=True)
    assert jnp.allclose(output, repeat)
