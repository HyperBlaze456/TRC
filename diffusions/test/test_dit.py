"""Basic integration tests for the Diffusion Transformer."""

import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
pytest.importorskip("flax")

from flax import nnx

from diffusions.utils.dit import DiT


def test_dit_forward_deterministic():
    rng = jax.random.key(0)
    rngs = nnx.Rngs(params=rng)

    model = DiT(
        in_features=8,
        hidden_size=16,
        depth=2,
        num_heads=4,
        rngs=rngs,
        conditioning_dim=12,
    )

    x_key, cond_key = jax.random.split(jax.random.key(1))
    x = jax.random.normal(x_key, (2, 6, 8))
    timesteps = jnp.array([10.0, 50.0])
    conditioning = jax.random.normal(cond_key, (2, 12))
    mask = jnp.ones((2, 6, 6))

    output = model(x, timesteps=timesteps, conditioning=conditioning, mask=mask, deterministic=True)

    assert output.shape == x.shape
    assert jnp.isfinite(output).all()

    repeat = model(x, timesteps=timesteps, conditioning=conditioning, mask=mask, deterministic=True)
    assert jnp.allclose(output, repeat)
