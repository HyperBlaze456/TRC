import jax
import jax.numpy as jnp
import optax
from functools import partial

def lsgan_loss(
        real_outputs: list[jax.Array], fake_outputs: list[jax.Array]
) -> jax.Array:

    total_loss = 0.0

    for real_out, fake_out in zip(real_outputs, fake_outputs):
        real_loss = jnp.mean(jnp.square(real_out - 1))
        fake_loss = jnp.mean(jnp.square(fake_out))
        total_loss += real_loss + fake_loss

    return total_loss / len(real_outputs)

def hinge_loss(
        real_outputs: list[jax.Array], fake_outputs: list[jax.Array]
) -> jax.Array:
    total_loss = 0.0
    for real_out, fake_out in zip(real_outputs, fake_outputs):
        real_loss = jnp.mean(optax.hinge_loss(real_out, jnp.ones_like(real_out)))
        fake_loss = jnp.mean(optax.hinge_loss(fake_out, -jnp.ones_like(fake_out)))

        total_loss += real_loss + fake_loss

    return total_loss / len(real_outputs)


def compute_discriminator_loss(
    disc_outputs_real: list[jax.Array],
    disc_outputs_fake: list[jax.Array],
    loss_type: str = "lsgan",
) -> jax.Array:
    """Compute discriminator loss.

    Args:
        disc_outputs_real: Discriminator outputs for real samples
        disc_outputs_fake: Discriminator outputs for fake samples
        loss_type: "lsgan" or "hinge"

    Returns:
        total_loss: Scalar discriminator loss
    """
    if loss_type == "lsgan":
        d_loss = lsgan_loss(disc_outputs_real, disc_outputs_fake)
    elif loss_type == "hinge":
        d_loss = hinge_loss(disc_outputs_real, disc_outputs_fake)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

    return d_loss


compute_discriminator_loss_lsgan = partial(
    compute_discriminator_loss, loss_type="lsgan"
)
compute_discriminator_loss_hinge = partial(
    compute_discriminator_loss, loss_type="hinge"
)
