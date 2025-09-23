from __future__ import annotations

from typing import Literal

import jax
import jax.numpy as jnp
from flax import nnx

from dit import DiT

TimeSchedule = Literal["linear", "cosine"]


def sample_timesteps(
    key: jax.Array,
    batch_size: int,
    *,
    schedule: TimeSchedule = "linear",
) -> jax.Array:
    """Sample time values in [0, 1] according to the requested schedule."""
    t = jax.random.uniform(key, (batch_size,), minval=0.0, maxval=1.0)
    if schedule == "linear":
        return t
    if schedule == "cosine":
        return 1.0 - jnp.cos(t * 0.5 * jnp.pi)
    raise ValueError(f"Unsupported time schedule: {schedule}")


def time_schedule(num_steps: int, *, schedule: TimeSchedule = "linear") -> jax.Array:
    """Deterministic time grid including the start and end points."""
    if num_steps <= 0:
        raise ValueError("num_steps must be positive")
    grid = jnp.linspace(0.0, 1.0, num_steps + 1)
    if schedule == "linear":
        return grid
    if schedule == "cosine":
        return 1.0 - jnp.cos(grid * 0.5 * jnp.pi)
    raise ValueError(f"Unsupported time schedule: {schedule}")


def _expand_timesteps(timesteps: jax.Array, ndim: int) -> jax.Array:
    if timesteps.ndim == 1:
        shape = (timesteps.shape[0],) + (1,) * (ndim - 1)
        return timesteps.reshape(shape)
    if timesteps.ndim == 2 and timesteps.shape[1] == 1:
        shape = (timesteps.shape[0],) + (1,) * (ndim - 1)
        return timesteps.reshape(shape)
    raise ValueError("timesteps must have shape [batch] or [batch, 1]")


def linear_interpolate(
    noise: jax.Array,
    data: jax.Array,
    timesteps: jax.Array,
    *,
    sigma_min: float = 1e-4,
) -> tuple[jax.Array, jax.Array]:
    """Linear interpolation used by flow matching.

    Returns the interpolant y(t) and the target velocity direction u.
    """
    if noise.shape != data.shape:
        raise ValueError("noise and data must have the same shape")
    if noise.shape[0] != timesteps.shape[0]:
        raise ValueError("Batch dimension of timesteps must match inputs")

    t = _expand_timesteps(timesteps, noise.ndim)
    scaling = 1.0 - (1.0 - sigma_min) * t
    interpolant = scaling * noise + t * data
    target_velocity = data - (1.0 - sigma_min) * noise
    return interpolant, target_velocity


def classifier_free_dropout(
    conditioning: jax.Array | None,
    *,
    rate: float,
    key: jax.Array,
) -> jax.Array | None:
    """Apply classifier-free guidance dropout to conditioning inputs."""
    if conditioning is None or rate <= 0.0:
        return conditioning
    if rate >= 1.0:
        return jnp.zeros_like(conditioning)
    keep_prob = 1.0 - rate
    broadcast_shape = (conditioning.shape[0],) + (1,) * (conditioning.ndim - 1)
    mask = jax.random.bernoulli(key, keep_prob, broadcast_shape)
    return conditioning * mask.astype(conditioning.dtype)


def _masked_mse(
    prediction: jax.Array,
    target: jax.Array,
    mask: jax.Array | None,
) -> jax.Array:
    error = (prediction - target) ** 2
    if mask is None:
        return jnp.mean(error)
    if mask.shape != prediction.shape:
        raise ValueError("Mask must have the same shape as prediction")
    total_weight = jnp.sum(mask)
    total_weight = jnp.maximum(total_weight, 1e-12)
    return jnp.sum(error * mask) / total_weight


class ConditionalFlowMatcher(nnx.Module):
    """Velocity estimator for conditional flow matching based on DiT."""

    def __init__(
        self,
        *,
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
        self.backbone = DiT(
            in_features=in_features,
            hidden_size=hidden_size,
            depth=depth,
            num_heads=num_heads,
            rngs=rngs,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            conditioning_dim=conditioning_dim,
            use_pos_emb=use_pos_emb,
        )

    def __call__(
        self,
        sample: jax.Array,
        timesteps: jax.Array,
        *,
        conditioning: jax.Array | None = None,
        mask: jax.Array | None = None,
        deterministic: bool = False,
    ) -> jax.Array:
        return self.backbone(
            sample,
            timesteps=timesteps,
            conditioning=conditioning,
            mask=mask,
            deterministic=deterministic,
        )


def flow_matching_loss(
    model: ConditionalFlowMatcher,
    noise: jax.Array,
    data: jax.Array,
    timesteps: jax.Array,
    *,
    sigma_min: float = 1e-4,
    conditioning: jax.Array | None = None,
    mask: jax.Array | None = None,
    cond_dropout_rate: float = 0.0,
    dropout_key: jax.Array | None = None,
    deterministic: bool = False,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Compute the masked MSE flow matching loss."""
    interpolant, target_velocity = linear_interpolate(
        noise, data, timesteps, sigma_min=sigma_min
    )

    if cond_dropout_rate > 0.0:
        if dropout_key is None:
            raise ValueError("dropout_key must be provided when cond_dropout_rate > 0")
        conditioning = classifier_free_dropout(
            conditioning, rate=cond_dropout_rate, key=dropout_key
        )

    predicted_velocity = model(
        interpolant,
        timesteps,
        conditioning=conditioning,
        mask=mask,
        deterministic=deterministic,
    )

    loss = _masked_mse(predicted_velocity, target_velocity, mask)
    aux = {
        "interpolant": interpolant,
        "target_velocity": target_velocity,
        "predicted_velocity": predicted_velocity,
    }
    return loss, aux


def euler_integrate(
    model: ConditionalFlowMatcher,
    initial: jax.Array,
    times: jax.Array,
    *,
    conditioning: jax.Array | None = None,
    mask: jax.Array | None = None,
    deterministic: bool = True,
) -> jax.Array:
    """Solve the flow ODE with Euler integration using a fixed time grid."""
    if times.ndim != 1:
        raise ValueError("times must be a 1D array")
    if times.shape[0] < 2:
        raise ValueError("times must contain at least a start and end point")

    batch_size = initial.shape[0]

    def body(step: int, carry: jax.Array) -> jax.Array:
        t0 = times[step]
        t1 = times[step + 1]
        dt = t1 - t0
        t_vec = jnp.full((batch_size,), t0, dtype=times.dtype)
        velocity = model(
            carry,
            t_vec,
            conditioning=conditioning,
            mask=mask,
            deterministic=deterministic,
        )
        return carry + dt * velocity

    num_steps = times.shape[0] - 1
    return jax.lax.fori_loop(0, num_steps, body, initial)
