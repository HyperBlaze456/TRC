"""Tests for conditional flow matching utilities."""

import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
pytest.importorskip("flax")

from flax import nnx

from diffusions.utils.flow_matching import (
    ConditionalFlowMatcher,
    classifier_free_dropout,
    euler_integrate,
    flow_matching_loss,
    linear_interpolate,
    sample_timesteps,
    time_schedule,
)


def test_sample_and_schedule_shapes():
    key = jax.random.key(0)
    batch = 4
    samples = sample_timesteps(key, batch)
    assert samples.shape == (batch,)
    assert jnp.all(samples >= 0.0) and jnp.all(samples <= 1.0)

    steps = time_schedule(5)
    assert steps.shape == (6,)
    assert jnp.allclose(steps[0], 0.0)
    assert jnp.allclose(steps[-1], 1.0)


def test_linear_interpolate_matches_expectations():
    noise = jnp.zeros((2, 3, 1))
    data = jnp.ones((2, 3, 1))
    t = jnp.array([0.0, 1.0])
    interpolant, velocity = linear_interpolate(noise, data, t, sigma_min=0.1)

    assert interpolant.shape == noise.shape
    assert velocity.shape == noise.shape

    assert jnp.allclose(interpolant[0], noise[0])
    assert jnp.allclose(interpolant[1], data[1])
    assert jnp.allclose(velocity, jnp.ones_like(velocity))


def test_classifier_free_dropout_zero_rate_identity():
    key = jax.random.key(1)
    cond = jnp.ones((2, 4))
    dropped = classifier_free_dropout(cond, rate=0.0, key=key)
    assert jnp.allclose(dropped, cond)

    dropped_full = classifier_free_dropout(cond, rate=1.0, key=key)
    assert jnp.all(dropped_full == 0)


def test_flow_matching_loss_runs_and_returns_aux():
    rng = jax.random.key(2)
    rngs = nnx.Rngs(params=rng)
    model = ConditionalFlowMatcher(
        in_features=4,
        hidden_size=8,
        depth=1,
        num_heads=2,
        rngs=rngs,
    )

    key_base = jax.random.key(3)
    key_noise, key_data, key_t = jax.random.split(key_base, 3)
    noise = jax.random.normal(key_noise, (2, 5, 4))
    data = jax.random.normal(key_data, (2, 5, 4))
    timesteps = sample_timesteps(key_t, 2)
    mask = jnp.ones_like(noise)

    loss, aux = flow_matching_loss(
        model,
        noise,
        data,
        timesteps,
        mask=mask,
        deterministic=True,
    )

    assert loss.shape == ()
    assert jnp.isfinite(loss)
    assert set(aux.keys()) == {
        "interpolant",
        "target_velocity",
        "predicted_velocity",
    }
    for value in aux.values():
        assert value.shape == noise.shape


def test_euler_integrate_shape_and_determinism():
    rng = jax.random.key(4)
    rngs = nnx.Rngs(params=rng)
    model = ConditionalFlowMatcher(
        in_features=4,
        hidden_size=8,
        depth=1,
        num_heads=2,
        rngs=rngs,
    )

    init = jax.random.normal(jax.random.key(5), (2, 3, 4))
    schedule = time_schedule(3)

    result = euler_integrate(model, init, schedule, deterministic=True)
    assert result.shape == init.shape

    repeat = euler_integrate(model, init, schedule, deterministic=True)
    assert jnp.allclose(result, repeat)
