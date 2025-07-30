from functools import partial

import jax
import jax.numpy as jnp

from tokenizer.utils.mel import MelSpectrogramJAX

def l1_loss(predictions: jax.Array, targets: jax.Array, mask: jax.Array) -> jax.Array:
    """L1 loss with masking.

    Args:
        predictions: Predicted audio [B, T, C]
        targets: Target audio [B, T, C]
        mask: Padding mask [B, T] where True = valid

    Returns:
        Scalar L1 loss
    """
    # Expand mask to match audio shape
    mask = jnp.expand_dims(mask, axis=-1)  # [B, T, 1]

    loss = jnp.abs(predictions - targets)
    loss = loss * mask
    return jnp.sum(loss) / jnp.maximum(jnp.sum(mask), 1.0)