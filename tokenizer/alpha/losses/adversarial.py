import jax
import jax.numpy as jnp
from typing import List, Tuple, Union


def hinge_loss_generator(disc_outputs_fake: Union[jnp.ndarray, List[jnp.ndarray]]) -> jnp.ndarray:
    """Hinge loss for generator.
    
    The generator tries to make the discriminator output positive values for fake samples.
    
    Args:
        disc_outputs_fake: Discriminator outputs for generated samples.
            Can be a single array or list of arrays (for multi-scale discriminators).
    
    Returns:
        Scalar loss value.
    """
    if isinstance(disc_outputs_fake, list):
        losses = [-jnp.mean(d) for d in disc_outputs_fake]
        return jnp.mean(jnp.array(losses))
    else:
        return -jnp.mean(disc_outputs_fake)


def hinge_loss_discriminator(
    disc_outputs_real: Union[jnp.ndarray, List[jnp.ndarray]], 
    disc_outputs_fake: Union[jnp.ndarray, List[jnp.ndarray]]
) -> jnp.ndarray:
    """Hinge loss for discriminator.
    
    The discriminator tries to output positive values for real samples 
    and negative values for fake samples.
    
    Args:
        disc_outputs_real: Discriminator outputs for real samples.
        disc_outputs_fake: Discriminator outputs for generated samples.
    
    Returns:
        Scalar loss value.
    """
    if isinstance(disc_outputs_real, list):
        real_losses = [jnp.mean(jax.nn.relu(1 - d)) for d in disc_outputs_real]
        fake_losses = [jnp.mean(jax.nn.relu(1 + d)) for d in disc_outputs_fake]
        real_loss = jnp.mean(jnp.array(real_losses))
        fake_loss = jnp.mean(jnp.array(fake_losses))
    else:
        real_loss = jnp.mean(jax.nn.relu(1 - disc_outputs_real))
        fake_loss = jnp.mean(jax.nn.relu(1 + disc_outputs_fake))
    
    return real_loss + fake_loss


def least_squares_loss_generator(disc_outputs_fake: Union[jnp.ndarray, List[jnp.ndarray]]) -> jnp.ndarray:
    """Least squares loss for generator (LSGAN).
    
    The generator tries to make the discriminator output 1 for fake samples.
    
    Args:
        disc_outputs_fake: Discriminator outputs for generated samples.
    
    Returns:
        Scalar loss value.
    """
    if isinstance(disc_outputs_fake, list):
        losses = [jnp.mean((d - 1) ** 2) for d in disc_outputs_fake]
        return jnp.mean(jnp.array(losses))
    else:
        return jnp.mean((disc_outputs_fake - 1) ** 2)


def least_squares_loss_discriminator(
    disc_outputs_real: Union[jnp.ndarray, List[jnp.ndarray]], 
    disc_outputs_fake: Union[jnp.ndarray, List[jnp.ndarray]]
) -> jnp.ndarray:
    """Least squares loss for discriminator (LSGAN).
    
    The discriminator tries to output 1 for real samples and 0 for fake samples.
    
    Args:
        disc_outputs_real: Discriminator outputs for real samples.
        disc_outputs_fake: Discriminator outputs for generated samples.
    
    Returns:
        Scalar loss value.
    """
    if isinstance(disc_outputs_real, list):
        real_losses = [jnp.mean((d - 1) ** 2) for d in disc_outputs_real]
        fake_losses = [jnp.mean(d ** 2) for d in disc_outputs_fake]
        real_loss = jnp.mean(jnp.array(real_losses))
        fake_loss = jnp.mean(jnp.array(fake_losses))
    else:
        real_loss = jnp.mean((disc_outputs_real - 1) ** 2)
        fake_loss = jnp.mean(disc_outputs_fake ** 2)
    
    return 0.5 * (real_loss + fake_loss)


def feature_matching_loss(
    features_real: Union[List[jnp.ndarray], List[List[jnp.ndarray]]],
    features_fake: Union[List[jnp.ndarray], List[List[jnp.ndarray]]]
) -> jnp.ndarray:
    """Feature matching loss for improved perceptual quality.
    
    Matches intermediate discriminator features between real and generated samples.
    This helps the generator produce samples with similar statistics to real data.
    
    Args:
        features_real: Intermediate features from discriminator for real samples.
            Can be a list (single discriminator) or list of lists (multi-scale).
        features_fake: Intermediate features from discriminator for generated samples.
    
    Returns:
        Scalar loss value.
    """
    if isinstance(features_real[0], list):
        # Multi-scale discriminator case
        losses = []
        for feat_real_scale, feat_fake_scale in zip(features_real, features_fake):
            scale_losses = []
            for feat_real, feat_fake in zip(feat_real_scale, feat_fake_scale):
                scale_losses.append(jnp.mean(jnp.abs(feat_real - feat_fake)))
            losses.append(jnp.mean(jnp.array(scale_losses)))
        return jnp.mean(jnp.array(losses))
    else:
        # Single discriminator case
        losses = []
        for feat_real, feat_fake in zip(features_real, features_fake):
            losses.append(jnp.mean(jnp.abs(feat_real - feat_fake)))
        return jnp.mean(jnp.array(losses))


def gradient_penalty(
    discriminator_fn,
    real_samples: jnp.ndarray,
    fake_samples: jnp.ndarray,
    rng_key: jax.random.PRNGKey,
    lambda_gp: float = 10.0
) -> jnp.ndarray:
    """Gradient penalty for improved training stability (WGAN-GP).
    
    Args:
        discriminator_fn: Discriminator function that takes samples and returns outputs.
        real_samples: Real audio samples.
        fake_samples: Generated audio samples.
        rng_key: JAX random key.
        lambda_gp: Gradient penalty coefficient.
    
    Returns:
        Scalar gradient penalty loss.
    """
    batch_size = real_samples.shape[0]
    alpha = jax.random.uniform(rng_key, shape=(batch_size, 1, 1))
    
    # Interpolate between real and fake samples
    interpolated = real_samples * alpha + fake_samples * (1 - alpha)
    
    # Compute gradients with respect to interpolated samples
    def disc_fn(x):
        outputs = discriminator_fn(x)
        # Handle multi-scale discriminator outputs
        if isinstance(outputs, list):
            return jnp.sum(jnp.array([jnp.sum(o) for o in outputs]))
        return jnp.sum(outputs)
    
    gradients = jax.grad(disc_fn)(interpolated)
    gradients_norm = jnp.sqrt(jnp.sum(gradients ** 2, axis=(1, 2)) + 1e-8)
    
    # Penalize gradients that deviate from 1
    gradient_penalty = jnp.mean((gradients_norm - 1) ** 2)
    
    return lambda_gp * gradient_penalty