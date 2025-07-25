"""
Discriminator losses for audio tokenizer GAN training.

This module implements LSGAN and Hinge losses for discriminator training.
All functions are designed to be JIT-compatible for efficient training.
"""

import jax
import jax.numpy as jnp
import optax
from functools import partial


# ============================================================================
# LSGAN Discriminator Loss
# ============================================================================

def lsgan_d_loss(
    real_outputs: list[jax.Array], 
    fake_outputs: list[jax.Array]
) -> jax.Array:
    """LSGAN discriminator loss.
    
    For LSGAN, discriminator tries to output 1 for real and 0 for fake.
    
    Args:
        real_outputs: List of discriminator outputs for real samples
        fake_outputs: List of discriminator outputs for generated samples
        
    Returns:
        Scalar discriminator loss
    """
    total_loss = 0.0
    
    for real_out, fake_out in zip(real_outputs, fake_outputs):
        # Discriminator wants to output 1 for real, 0 for fake
        real_loss = jnp.mean(jnp.square(real_out - 1))
        fake_loss = jnp.mean(jnp.square(fake_out))
        total_loss += real_loss + fake_loss
    
    return total_loss / len(real_outputs)


# ============================================================================
# Hinge Discriminator Loss
# ============================================================================

def hinge_d_loss(
    real_outputs: list[jax.Array], 
    fake_outputs: list[jax.Array]
) -> jax.Array:
    """Hinge discriminator loss using optax.
    
    For Hinge loss, discriminator tries to output >1 for real and <-1 for fake.
    
    Args:
        real_outputs: List of discriminator outputs for real samples
        fake_outputs: List of discriminator outputs for generated samples
        
    Returns:
        Scalar discriminator loss
    """
    total_loss = 0.0
    
    for real_out, fake_out in zip(real_outputs, fake_outputs):
        # Using optax hinge loss
        # For real: we want output > 1, so target = 1, margin = 0
        real_loss = jnp.mean(optax.hinge_loss(real_out, jnp.ones_like(real_out)))
        
        # For fake: we want output < -1, so target = -1, margin = 0
        fake_loss = jnp.mean(optax.hinge_loss(fake_out, -jnp.ones_like(fake_out)))
        
        total_loss += real_loss + fake_loss
    
    return total_loss / len(real_outputs)


# ============================================================================
# Combined Discriminator Loss Function
# ============================================================================

def compute_discriminator_loss(
    disc_outputs_real: list[jax.Array],
    disc_outputs_fake: list[jax.Array], 
    loss_type: str = "lsgan"
) -> tuple[jax.Array, dict]:
    """Compute discriminator loss.
    
    Args:
        disc_outputs_real: Discriminator outputs for real samples
        disc_outputs_fake: Discriminator outputs for fake samples
        loss_type: "lsgan" or "hinge"
        
    Returns:
        total_loss: Scalar discriminator loss
        metrics: Dictionary of loss components
    """
    # Adversarial loss
    if loss_type == "lsgan":
        d_loss = lsgan_d_loss(disc_outputs_real, disc_outputs_fake)
    elif loss_type == "hinge":
        d_loss = hinge_d_loss(disc_outputs_real, disc_outputs_fake)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
    
    # Compute average predictions for monitoring
    avg_real = jnp.mean(jnp.concatenate([out.flatten() for out in disc_outputs_real]))
    avg_fake = jnp.mean(jnp.concatenate([out.flatten() for out in disc_outputs_fake]))
    
    metrics = {
        'd_loss': d_loss,
        'd_real_avg': avg_real,
        'd_fake_avg': avg_fake,
    }
    
    return d_loss, metrics


# ============================================================================
# JIT-compatible versions using partial for static arguments
# ============================================================================

# Discriminator loss functions with static loss_type  
compute_discriminator_loss_lsgan = partial(compute_discriminator_loss, loss_type="lsgan")
compute_discriminator_loss_hinge = partial(compute_discriminator_loss, loss_type="hinge")