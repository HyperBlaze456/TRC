import jax
import jax.numpy as jnp
import optax


def compute_ctc_loss(
    logits: jax.Array,
    logit_paddings: jax.Array,
    labels: jax.Array,
    label_paddings: jax.Array,
    blank_id: int = 0,
    log_epsilon: float = -100000.0,
) -> jax.Array:
    """Compute CTC loss for phoneme prediction.

    Args:
        logits: Model predictions with shape [B, T, K] where:
            B = batch size
            T = encoder output time steps (e.g., 50Hz frames)
            K = phoneme vocabulary size (218 including blank)
        logit_paddings: Padding mask for logits [B, T] where 1.0 = padded
        labels: Target phoneme indices [B, N] where N = max phoneme sequence length
        label_paddings: Padding mask for labels [B, N] where 1.0 = padded
        blank_id: Index of blank token (default: 0)
        log_epsilon: Small value for numerical stability

    Returns:
        ctc_losses: Per-sequence CTC loss [B]
    """
    # Compute CTC loss using optax
    ctc_losses = optax.ctc_loss(
        logits=logits,
        logit_paddings=logit_paddings,
        labels=labels,
        label_paddings=label_paddings,
        blank_id=blank_id,
        log_epsilon=log_epsilon,
    )

    return ctc_losses


def phoneme_ctc_loss(
    phoneme_logits: jax.Array,
    encoder_mask: jax.Array,
    phoneme_indices: jax.Array,
    phoneme_mask: jax.Array,
    blank_id: int = 0,
    reduction: str = "mean",
) -> jax.Array:
    """Compute phoneme CTC loss with proper masking.

    This function handles the conversion from encoder masks to CTC-compatible
    padding format and computes the loss.

    Args:
        phoneme_logits: Phoneme predictions from model [B, T, K] where:
            - B = batch size
            - T = encoder output length(sequence len)
            - K = phoneme vocabulary size, sum of 1.0
        encoder_mask: Encoder validity mask [B, T] where True = valid
        phoneme_indices: Target phoneme indices [B, N]
        phoneme_mask: Phoneme padding mask [B, N] where 1.0 = padded (CTC format)
        blank_id: Index of blank token. Don't change this.
        reduction: How to reduce batch losses ('mean', 'sum', or 'none')

    Returns:
        loss: Scalar loss (if reduction != 'none') or per-sequence losses [B]
    """
    # Convert encoder mask to CTC padding format
    # CTC requires [B, T] with **1.0 = padded**.
    logit_paddings = 1.0 - encoder_mask.astype(jnp.float32)  # Invert: 1.0 = padded

    # Compute CTC loss
    ctc_losses = compute_ctc_loss(
        logits=phoneme_logits,
        logit_paddings=logit_paddings,
        labels=phoneme_indices,
        label_paddings=phoneme_mask,
        blank_id=blank_id,  # 0
    )

    # Apply reduction
    if reduction == "mean":
        loss = jnp.mean(ctc_losses)
    elif reduction == "sum":
        loss = jnp.sum(ctc_losses)
    else:  # 'none'
        loss = ctc_losses

    return loss
